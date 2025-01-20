import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from google_titan_implementation import TitanAttention


# First, let's implement the Memory as Context variant
class MemoryAsContextTitan(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=64,
        chunk_size=256,
        num_persistent_memory=32,
        memory_size=1024,
        dropout=0.1
    ):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.num_persistent_memory = num_persistent_memory
        
        # Persistent memory parameters
        self.persistent_memory = nn.Parameter(torch.randn(num_persistent_memory, dim))
        
        # Query projection for memory retrieval
        self.query_proj = nn.Linear(dim, dim)
        
        # Long-term memory module
        self.memory = NeuralMemory(memory_size, dim)
        
        # Main attention module
        self.attention = TitanAttention(dim, num_heads, head_dim, dropout)
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Chunk the input sequence
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        chunks = torch.split(x, self.chunk_size, dim=1)
        
        outputs = []
        memory_state = self.memory.init_state(batch_size)
        
        for chunk in chunks:
            # Generate query for memory retrieval
            query = self.query_proj(chunk)
            
            # Retrieve historical information from memory
            historical_info = self.memory.retrieve(memory_state, query)
            
            # Combine persistent memory, historical info, and current chunk
            persistent_memory_expanded = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
            combined_input = torch.cat([
                persistent_memory_expanded,
                historical_info,
                chunk
            ], dim=1)
            
            # Apply attention
            attended = self.attention(combined_input)
            
            # Update memory state
            memory_state = self.memory.update(memory_state, attended)
            
            # Generate final output
            memory_output = self.memory.retrieve(memory_state, attended)
            output = attended * memory_output
            
            outputs.append(output[:, -chunk.size(1):])  # Only keep chunk-corresponding outputs
        
        # Concatenate all outputs
        return torch.cat(outputs, dim=1)

# Neural Memory implementation
class NeuralMemory(nn.Module):
    def __init__(self, memory_size, dim):
        super().__init__()
        self.memory_size = memory_size
        self.dim = dim
        
        # Memory operations
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.query_proj = nn.Linear(dim, dim)
        
    def init_state(self, batch_size):
        return torch.zeros(batch_size, self.memory_size, self.dim)
    
    def retrieve(self, memory_state, query):
        # Project query
        query = self.query_proj(query)
        
        # Compute attention scores
        keys = self.key_proj(memory_state)
        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.dim)
        attention = F.softmax(scores, dim=-1)
        
        # Retrieve values
        values = self.value_proj(memory_state)
        retrieved = torch.matmul(attention, values)
        
        return retrieved
    
    def update(self, memory_state, input_data):
        # Simple update mechanism - could be made more sophisticated
        return 0.9 * memory_state + 0.1 * input_data

# Now let's implement the Gated Memory variant
class GatedMemoryTitan(nn.Module):
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
        
        # Memory module
        self.memory = NeuralMemory(memory_size, dim)
        
        # Sliding window attention
        self.sliding_attention = SlidingWindowAttention(dim, num_heads, head_dim, window_size, dropout)
        
        # Gating mechanism
        self.gate_norm1 = nn.LayerNorm(dim)
        self.gate_norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        
        # Combine input with persistent memory
        persistent_memory_expanded = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
        combined_input = torch.cat([persistent_memory_expanded, x], dim=1)
        
        # Process through sliding window attention
        attention_output = self.sliding_attention(combined_input, mask)
        
        # Process through memory
        memory_output = self.memory.retrieve(
            self.memory.update(self.memory.init_state(batch_size), combined_input),
            combined_input
        )
        
        # Apply gating mechanism
        normalized_attention = self.gate_norm1(attention_output)
        normalized_memory = self.gate_norm2(memory_output)
        gate_input = torch.cat([normalized_attention, normalized_memory], dim=-1)
        gate_weights = self.gate(gate_input)
        
        output = gate_weights * normalized_attention + (1 - gate_weights) * normalized_memory
        
        return output

# Sliding Window Attention implementation
class SlidingWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, window_size, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        
        self.attention = TitanAttention(dim, num_heads, head_dim, dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Create sliding window attention mask
        window_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            window_start = max(0, i - self.window_size)
            window_end = min(seq_len, i + 1)
            window_mask[i, window_start:window_end] = False
        
        window_mask = window_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        if mask is not None:
            window_mask = window_mask | (~mask.bool())
        
        # Apply attention with sliding window mask
        return self.attention(x, window_mask)