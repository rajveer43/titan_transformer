import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from typing import List, Dict
import re

# Depthwise Separable Convolution
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Gating Mechanism
class GatingMechanism(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim)
        self.transform_proj = nn.Linear(dim, dim)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        transformed = self.transform_proj(x)
        return gate * transformed

# Memory Module
class MemoryModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.memory = nn.Parameter(torch.zeros(dim, dim))

    def update_memory(self, key, value, forgetting_rate):
        self.memory = (1 - forgetting_rate) * self.memory + torch.matmul(key.T, value)

    def retrieve_memory(self, query):
        return torch.matmul(query, self.memory)

# Titan Attention
class TitanAttention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=64, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Projections for Q, K, V
        self.q_proj = nn.Linear(dim, num_heads * head_dim)
        self.k_proj = nn.Linear(dim, num_heads * head_dim)
        self.v_proj = nn.Linear(dim, num_heads * head_dim)

        # Depthwise separable convolutions
        self.q_conv = DepthwiseSeparableConv1d(num_heads * head_dim, num_heads * head_dim, kernel_size=3, padding=1)
        self.k_conv = DepthwiseSeparableConv1d(num_heads * head_dim, num_heads * head_dim, kernel_size=3, padding=1)
        self.v_conv = DepthwiseSeparableConv1d(num_heads * head_dim, num_heads * head_dim, kernel_size=3, padding=1)

        self.out_proj = nn.Linear(num_heads * head_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape

        # Project and apply SiLU activation
        q = F.silu(self.q_proj(x))
        k = F.silu(self.k_proj(x))
        v = F.silu(self.v_proj(x))

        # Reshape for depthwise conv
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply depthwise separable convolutions
        q = self.q_conv(q).transpose(1, 2)
        k = self.k_conv(k).transpose(1, 2)
        v = self.v_conv(v).transpose(1, 2)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project output
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.out_proj(out)

        return out

# Titan Block
class TitanBlock(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=64, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TitanAttention(dim, num_heads, head_dim, dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            GatingMechanism(mlp_dim),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Residual connection for attention
        x = x + self.attn(self.norm1(x), mask)
        # Residual connection for MLP
        x = x + self.mlp(self.norm2(x))
        return x

# Titan Transformer
class TitanTransformer(nn.Module):
    def __init__(self, dim, depth, num_heads=8, head_dim=64, mlp_ratio=4, dropout=0.1, num_persistent_tokens=4):
        super().__init__()
        self.persistent_memory = nn.Parameter(torch.zeros(1, num_persistent_tokens, dim))
        self.layers = nn.ModuleList([
            TitanBlock(dim, num_heads, head_dim, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        x = torch.cat([self.persistent_memory.expand(batch_size, -1, -1), x], dim=1)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Custom Tokenizer (Simplified)
class CustomTokenizer:
    def __init__(self, vocab_size=50000, min_freq=2, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens
        self.token2idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.idx2token = {idx: token for idx, token in enumerate(special_tokens)}
        self.vocab_size_current = len(special_tokens)
        self.token_pattern = re.compile(r'\w+|[^\w\s]')

    def train_from_texts(self, texts):
        word_counts = Counter()
        for text in texts:
            tokens = self._basic_tokenize(text)
            word_counts.update(tokens)

        filtered_tokens = [
            token for token, count in word_counts.most_common()
            if count >= self.min_freq and token not in self.special_tokens
        ]

        for token in filtered_tokens[:self.vocab_size - len(self.special_tokens)]:
            self.token2idx[token] = self.vocab_size_current
            self.idx2token[self.vocab_size_current] = token
            self.vocab_size_current += 1

    def _basic_tokenize(self, text):
        return self.token_pattern.findall(text.lower())

    def encode(self, text, add_special_tokens=True):
        tokens = self._basic_tokenize(text)
        ids = [self.token2idx.get(token, self.token2idx["<UNK>"]) for token in tokens]
        if add_special_tokens:
            ids = [self.token2idx["<BOS>"]] + ids + [self.token2idx["<EOS>"]]
        return ids

    def decode(self, ids):
        return " ".join(self.idx2token.get(idx, "<UNK>") for idx in ids)

# Example Usage
if __name__ == "__main__":
    tokenizer = CustomTokenizer()
    texts = ["This is a sample text.", "Another example text."]
    tokenizer.train_from_texts(texts)

    model = TitanTransformer(dim=512, depth=6, num_heads=8, head_dim=64, mlp_ratio=4, dropout=0.1)
    sample_input = torch.randint(0, 100, (2, 50))  # Batch size of 2, sequence length of 50
    output = model(sample_input)
    print(output.shape)
