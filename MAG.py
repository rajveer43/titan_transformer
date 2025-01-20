import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from google_titan_implementation import TitanAttention
from MAC import SlidingWindowAttention, MemoryAsContextTitan, NeuralMemory




class MemoryAsGateTitan(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=64,
        window_size=256,
        num_persistent_memory=32,
        memory_size=1024,
        dropout=0.1
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_persistent_memory = num_persistent_memory
        
        # Persistent memory parameters
        self.persistent_memory = nn.Parameter(torch.randn(num_persistent_memory, dim))
        
        # Memory module branch
        self.memory = NeuralMemory(memory_size, dim)
        
        # Sliding window attention branch with prefix support
        self.sliding_attention = SlidingWindowAttentionWithPrefix(
            dim, 
            num_heads, 
            head_dim, 
            window_size, 
            dropout
        )
        
        # Normalization layers for gating
        self.norm_memory = nn.LayerNorm(dim)
        self.norm_attention = nn.LayerNorm(dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        
        # Add persistent memory as prefix
        persistent_memory_expanded = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
        prefixed_input = torch.cat([persistent_memory_expanded, x], dim=1)
        
        # Branch 1: Memory processing
        memory_state = self.memory.init_state(batch_size)
        memory_state = self.memory.update(memory_state, prefixed_input)
        memory_output = self.memory.retrieve(memory_state, prefixed_input)
        memory_output = self.norm_memory(memory_output)
        
        # Branch 2: Sliding window attention with prefix
        attention_output = self.sliding_attention(prefixed_input, mask)
        attention_output = self.norm_attention(attention_output)
        
        # Compute gating weights
        gate_input = torch.cat([memory_output, attention_output], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # Combine outputs using gating mechanism
        output = gate_weights * memory_output + (1 - gate_weights) * attention_output
        
        # Final processing
        output = self.dropout(self.output_proj(output))
        
        # Return only the non-prefix portion
        return output[:, self.num_persistent_memory:]

class SlidingWindowAttentionWithPrefix(nn.Module):
    def __init__(self, dim, num_heads, head_dim, window_size, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        
        self.attention = TitanAttention(dim, num_heads, head_dim, dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Create sliding window attention mask with prefix consideration
        window_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        
        # Allow full attention to prefix (persistent memory) tokens
        prefix_size = x.size(1) - mask.size(1) if mask is not None else 0
        window_mask[:, :prefix_size] = False
        
        # Create sliding window pattern for the rest
        for i in range(prefix_size, seq_len):
            window_start = max(prefix_size, i - self.window_size)
            window_end = min(seq_len, i + 1)
            window_mask[i, window_start:window_end] = False
        
        window_mask = window_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        if mask is not None:
            # Adjust mask to account for prefix tokens
            full_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
            full_mask[:, prefix_size:] = mask.bool()
            window_mask = window_mask | (~full_mask.unsqueeze(1))
        
        # Apply attention with sliding window mask
        return self.attention(x, window_mask)

# Example usage
def create_mag_model(
    dim=512,
    num_heads=8,
    head_dim=64,
    window_size=256,
    num_persistent_memory=32,
    memory_size=1024,
    dropout=0.1
):
    model = MemoryAsGateTitan(
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        window_size=window_size,
        num_persistent_memory=num_persistent_memory,
        memory_size=memory_size,
        dropout=dropout
    )
    return model