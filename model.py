from typing import List
import torch
import os
import json


import torch
import torch.nn as nn
from collections import Counter
from typing import List, Dict, Tuple
import re
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from ts.torch_handler.base_handler import BaseHandler
# from ts.utils.util import handler
from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



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


class GatingMechanism(nn.Module):
   def __init__(self, dim):
       super().__init__()
       self.gate_proj = nn.Linear(dim, dim)
       self.transform_proj = nn.Linear(dim, dim)


   def forward(self, x):
       gate = torch.sigmoid(self.gate_proj(x))
       transformed = self.transform_proj(x)
       return gate * transformed


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
       self.q_conv = DepthwiseSeparableConv1d(
           num_heads * head_dim,
           num_heads * head_dim,
           kernel_size=3,
           padding=1
       )
       self.k_conv = DepthwiseSeparableConv1d(
           num_heads * head_dim,
           num_heads * head_dim,
           kernel_size=3,
           padding=1
       )
       self.v_conv = DepthwiseSeparableConv1d(
           num_heads * head_dim,
           num_heads * head_dim,
           kernel_size=3,
           padding=1
       )


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


       # L2 normalize queries and keys
       q = F.normalize(q, p=2, dim=-1)
       k = F.normalize(k, p=2, dim=-1)


       # Compute attention scores
       attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale


       if mask is not None:
           # Properly reshape mask for broadcasting
           # mask shape: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
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


class TitanTransformer(nn.Module):
   def __init__(
       self,
       dim,
       depth,
       num_heads=8,
       head_dim=64,
       mlp_ratio=4,
       dropout=0.1
   ):
       super().__init__()
       self.layers = nn.ModuleList([
           TitanBlock(
               dim=dim,
               num_heads=num_heads,
               head_dim=head_dim,
               mlp_ratio=mlp_ratio,
               dropout=dropout
           )
           for _ in range(depth)
       ])


       self.norm = nn.LayerNorm(dim)


   def forward(self, x, mask=None):
       for layer in self.layers:
           x = layer(x, mask)
       return self.norm(x)


# Example usage
def create_titan_model(
   vocab_size=50000,
   max_seq_length=1024,
   dim=512,
   depth=12,
   num_heads=8,
   head_dim=64,
   mlp_ratio=4,
   dropout=0.1
):
   class TitanModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, dim)
           self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, dim))
           self.transformer = TitanTransformer(
               dim=dim,
               depth=depth,
               num_heads=num_heads,
               head_dim=head_dim,
               mlp_ratio=mlp_ratio,
               dropout=dropout
           )


       def forward(self, x, mask=None):
           # Add positional embeddings
           x = self.embedding(x)
           x = x + self.pos_embedding[:, :x.size(1), :]


           # Apply transformer
           x = self.transformer(x, mask)
           return x


   return TitanModel()


class CustomTokenizer:
   def __init__(
       self,
       vocab_size: int = 50000,
       min_freq: int = 2,
       special_tokens: List[str] = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
   ):
       self.vocab_size = vocab_size
       self.min_freq = min_freq
       self.special_tokens = special_tokens


       # Initialize special token IDs
       self.pad_token_id = 0
       self.unk_token_id = 1
       self.bos_token_id = 2
       self.eos_token_id = 3


       # Initialize vocabularies
       self.token2idx: Dict[str, int] = {token: idx for idx, token in enumerate(special_tokens)}
       self.idx2token: Dict[int, str] = {idx: token for idx, token in enumerate(special_tokens)}
       self.vocab_size_current = len(special_tokens)


       # Regex for tokenization
       self.token_pattern = re.compile(r'\w+|[^\w\s]')


   def train_from_texts(self, texts: List[str]) -> None:
       """Train tokenizer on a list of texts."""
       # Count word frequencies
       word_counts = Counter()


       for text in texts:
           tokens = self._basic_tokenize(text)
           word_counts.update(tokens)


       # Filter by minimum frequency and vocab size
       filtered_tokens = [
           token for token, count in word_counts.most_common()
           if count >= self.min_freq and token not in self.special_tokens
       ]


       # Add tokens to vocabulary up to vocab_size
       remaining_space = self.vocab_size - len(self.special_tokens)
       for token in filtered_tokens[:remaining_space]:
           self.token2idx[token] = self.vocab_size_current
           self.idx2token[self.vocab_size_current] = token
           self.vocab_size_current += 1


   def _basic_tokenize(self, text: str) -> List[str]:
       """Basic tokenization into words and punctuation."""
       return self.token_pattern.findall(text.lower())


   def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
       """Encode text to token ids."""
       tokens = self._basic_tokenize(text)


       ids = []
       if add_special_tokens:
           ids.append(self.bos_token_id)


       for token in tokens:
           ids.append(self.token2idx.get(token, self.unk_token_id))


       if add_special_tokens:
           ids.append(self.eos_token_id)


       return ids


   def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
       """Decode token ids back to text."""
       tokens = []
       for idx in ids:
           token = self.idx2token.get(idx, "<UNK>")
           if skip_special_tokens and token in self.special_tokens:
               continue
           tokens.append(token)
       return " ".join(tokens)


   def save_vocab(self, path: str) -> None:
       """Save vocabulary to file."""
       with open(path, 'w', encoding='utf-8') as f:
           for token, idx in sorted(self.token2idx.items(), key=lambda x: x[1]):
               f.write(f"{token}\t{idx}\n")


   def load_vocab(self, path: str) -> None:
       """Load vocabulary from file."""
       self.token2idx.clear()
       self.idx2token.clear()
       with open(path, 'r', encoding='utf-8') as f:
           for line in f:
               token, idx = line.strip().split('\t')
               idx = int(idx)
               self.token2idx[token] = idx
               self.idx2token[idx] = token
       self.vocab_size_current = len(self.token2idx)


class CustomEmbedding(nn.Module):
   def __init__(
       self,
       vocab_size: int,
       embedding_dim: int,
       pad_idx: int = 0,
       max_norm: float = None
   ):
       super().__init__()
       self.embedding = nn.Embedding(
           num_embeddings=vocab_size,
           embedding_dim=embedding_dim,
           padding_idx=pad_idx,
           max_norm=max_norm
       )
       self.embedding_dim = embedding_dim


       # Initialize embeddings using Xavier uniform initialization
       nn.init.xavier_uniform_(self.embedding.weight)
       with torch.no_grad():
           self.embedding.weight[pad_idx].fill_(0)


   def forward(self, x: torch.Tensor) -> torch.Tensor:
       return self.embedding(x)


class TextDataset(torch.utils.data.Dataset):
   def __init__(
       self,
       texts: List[str],
       tokenizer: CustomTokenizer,
       max_length: int = 1024
   ):
       self.tokenizer = tokenizer
       self.max_length = max_length


       # Tokenize all texts
       self.encoded_texts = [
           self.tokenizer.encode(text)[:max_length] for text in texts
       ]


   def __len__(self) -> int:
       return len(self.encoded_texts)


   def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
       tokens = self.encoded_texts[idx]


       # Create input and target sequences for language modeling
       input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
       target_ids = torch.tensor(tokens[1:], dtype=torch.long)


       return input_ids, target_ids


def create_dataloader(
   texts: List[str],
   tokenizer: CustomTokenizer,
   batch_size: int = 32,
   max_length: int = 1024,
   shuffle: bool = True
) -> torch.utils.data.DataLoader:
   dataset = TextDataset(texts, tokenizer, max_length)


   # Custom collate function to handle variable length sequences
   def collate_fn(batch):
       input_ids, target_ids = zip(*batch)


       # Pad sequences
       input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
       target_ids = pad_sequence(target_ids, batch_first=True, padding_value=tokenizer.pad_token_id)


       # Create attention mask
       attention_mask = (input_ids != tokenizer.pad_token_id).float()


       return {
           'input_ids': input_ids,
           'target_ids': target_ids,
           'attention_mask': attention_mask
       }


   return torch.utils.data.DataLoader(
       dataset,
       batch_size=batch_size,
       shuffle=shuffle,
       collate_fn=collate_fn
   )


# Example usage
def create_custom_tokenizer_and_embedding(
   texts: List[str],
   vocab_size: int = 50000,
   embedding_dim: int = 512,
   min_freq: int = 2
):
   # Create and train tokenizer
   tokenizer = CustomTokenizer(vocab_size=vocab_size, min_freq=min_freq)
   tokenizer.train_from_texts(texts)


   # Create embedding layer
   embedding = CustomEmbedding(
       vocab_size=tokenizer.vocab_size_current,
       embedding_dim=embedding_dim,
       pad_idx=tokenizer.pad_token_id
   )


   return tokenizer, embedding


class TitanModelWithCustomEmbedding(nn.Module):
   def __init__(
       self,
       embedding_layer: CustomEmbedding,
       max_seq_length: int = 1024,
       depth: int = 12,
       num_heads: int = 8,
       head_dim: int = 64,
       mlp_ratio: int = 4,
       dropout: float = 0.1
   ):
       super().__init__()
       dim = embedding_layer.embedding_dim


       self.embedding = embedding_layer
       self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, dim))


       self.transformer = TitanTransformer(
           dim=dim,
           depth=depth,
           num_heads=num_heads,
           head_dim=head_dim,
           mlp_ratio=mlp_ratio,
           dropout=dropout
       )


       # Add final projection for token prediction
       self.output_projection = nn.Linear(dim, embedding_layer.embedding.num_embeddings)


   def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
       # Get embeddings
       x = self.embedding(input_ids)


       # Add positional embeddings
       x = x + self.pos_embedding[:, :x.size(1), :]


       # Apply transformer
       x = self.transformer(x, attention_mask)


       # Project to vocabulary
       logits = self.output_projection(x)


       return logits


# Example usage
def create_complete_model(texts: List[str], vocab_size: int = 50000, embedding_dim: int = 512):
   # Create tokenizer and embedding
   tokenizer, embedding = create_custom_tokenizer_and_embedding(
       texts=texts,
       vocab_size=vocab_size,
       embedding_dim=embedding_dim
   )


   # Create model
   model = TitanModelWithCustomEmbedding(
       embedding_layer=embedding,
       max_seq_length=1024,
       depth=12
   )


   return model, tokenizer
