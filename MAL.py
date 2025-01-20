# Memory as Layer implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from google_titan_implementation import TitanAttention
from google_titan_implementation import TitansMemoryModule
from MAC import SlidingWindowAttention, MemoryAsContextTitan, NeuralMemory

class MemoryAsLayerTitan(nn.Module):
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
        
        # Memory module (Layer 1)
        self.memory = NeuralMemory(memory_size, dim)
        
        # Sliding window attention (Layer 2)
        self.sliding_attention = SlidingWindowAttention(dim, num_heads, head_dim, window_size, dropout)
        
        # Layer normalization for better stability
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        
        # Combine input with persistent memory
        persistent_memory_expanded = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
        combined_input = torch.cat([persistent_memory_expanded, x], dim=1)
        
        # Layer 1: Process through memory module
        memory_state = self.memory.init_state(batch_size)
        memory_state = self.memory.update(memory_state, combined_input)
        memory_output = self.memory.retrieve(memory_state, combined_input)
        memory_output = self.norm1(memory_output)
        
        # Layer 2: Process through sliding window attention
        attention_output = self.sliding_attention(memory_output, mask)
        attention_output = self.norm2(attention_output)
        
        # Final output projection
        output = self.output_proj(attention_output)
        
        return output

# Memory-only variant (without attention)
class MemoryOnlyTitan(nn.Module):
    def __init__(
        self,
        dim,
        num_persistent_memory=32,
        memory_size=1024,
        dropout=0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_persistent_memory = num_persistent_memory
        
        # Persistent memory parameters
        self.persistent_memory = nn.Parameter(torch.randn(num_persistent_memory, dim))
        
        # Memory module
        self.memory = NeuralMemory(memory_size, dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(dim)
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        
        # Combine input with persistent memory
        persistent_memory_expanded = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
        combined_input = torch.cat([persistent_memory_expanded, x], dim=1)
        
        # Process through memory module
        memory_state = self.memory.init_state(batch_size)
        memory_state = self.memory.update(memory_state, combined_input)
        memory_output = self.memory.retrieve(memory_state, combined_input)
        
        # Normalize and project output
        output = self.norm(memory_output)
        output = self.dropout(self.output_proj(output))
        
        return output