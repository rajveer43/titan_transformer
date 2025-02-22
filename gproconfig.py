import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

###############################
# 1. Define the Long-Term Memory Module
###############################
class LongTermMemoryModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_dim, num_layers=2):
        super(LongTermMemoryModule, self).__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.memory_update = nn.Linear(hidden_dim, memory_dim)
        # A learnable forget gate scalar (between 0 and 1 can be enforced via sigmoid if desired)
        self.forget_gate = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, prev_memory):
        # x: tensor of shape (batch, seq_len, input_dim)
        # prev_memory: tensor of shape (batch, memory_dim)
        # For simplicity, we average over the sequence to get a candidate update
        candidate = self.mlp(x)               # (batch, seq_len, hidden_dim)
        candidate = candidate.mean(dim=1)     # (batch, hidden_dim)
        candidate = self.memory_update(candidate)  # (batch, memory_dim)
        # Update memory: new_memory = forget_gate * prev_memory + (1 - forget_gate) * candidate
        new_memory = self.forget_gate * prev_memory + (1 - self.forget_gate) * candidate
        return new_memory

###############################
# 2. Define the Titans Architecture
###############################
class TitansModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, memory_dim, seq_length, persistent_mem_tokens):
        super(TitansModel, self).__init__()
        self.seq_length = seq_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # For simplicity, we use a Transformer encoder as our "core" branch.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # The long-term memory module learns to update a memory vector based on input “surprise”
        self.long_term_memory = LongTermMemoryModule(embed_dim, hidden_dim, memory_dim)
        # Persistent memory: a fixed set of learnable tokens (e.g., task-specific knowledge)
        self.persistent_memory = nn.Parameter(torch.randn(persistent_mem_tokens, embed_dim))
        # Final classifier that combines the transformer summary, long-term memory and persistent memory
        # Here, we flatten the persistent memory tokens into one long vector.
        final_input_dim = embed_dim + memory_dim + persistent_mem_tokens * embed_dim
        self.fc = nn.Linear(final_input_dim, vocab_size)
        
    def forward(self, input_ids, prev_memory):
        # input_ids: (batch, seq_length)
        # prev_memory: (batch, memory_dim)
        # Embed the input tokens
        x = self.embedding(input_ids)   # (batch, seq_length, embed_dim)
        # Transformer expects (seq_length, batch, embed_dim)
        x_trans = x.transpose(0, 1)       # (seq_length, batch, embed_dim)
        transformer_out = self.transformer(x_trans)  # (seq_length, batch, embed_dim)
        transformer_out = transformer_out.transpose(0, 1)  # (batch, seq_length, embed_dim)
        
        # For the memory update, we use the last token's representation as a query.
        query = transformer_out[:, -1, :]   # (batch, embed_dim)
        # Update the long-term memory based on the current query (we unsqueeze to simulate a one-step sequence)
        new_memory = self.long_term_memory(query.unsqueeze(1), prev_memory)
        
        # Aggregate transformer outputs; here we use mean pooling.
        transformer_summary = transformer_out.mean(dim=1)  # (batch, embed_dim)
        
        # Flatten persistent memory tokens and replicate for batch
        batch_size = input_ids.size(0)
        persistent_flat = self.persistent_memory.view(1, -1).repeat(batch_size, 1)  # (batch, persistent_mem_tokens*embed_dim)
        
        # Concatenate all features: transformer summary, long-term memory and persistent memory.
        combined = torch.cat([transformer_summary, new_memory, persistent_flat], dim=1)
        logits = self.fc(combined)  # (batch, vocab_size)
        return logits, new_memory

###############################
# 3. Define a GRPO (PPO-like) Loss Function
###############################
def grpo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2, kl_coeff=0.01):
    """
    Computes a PPO-style loss using the ratio of new to old log probabilities.
    For each sample:
      loss = -min(ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon) * advantage)
    A dummy KL penalty term is also added.
    """
    # Compute probability ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    # Dummy KL term (in practice compute actual KL divergence between policies)
    kl_loss = (new_log_probs - old_log_probs).mean()
    return pg_loss + kl_coeff * kl_loss

###############################
# 4. Training Loop Example
###############################
def train_titans_model():
    # Hyperparameters
    vocab_size = 10000
    embed_dim = 256
    num_heads = 8
    num_layers = 4
    hidden_dim = 512
    memory_dim = 256
    seq_length = 128
    persistent_mem_tokens = 10
    batch_size = 16
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TitansModel(vocab_size, embed_dim, num_heads, num_layers, hidden_dim, memory_dim, seq_length, persistent_mem_tokens)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # For illustration, we use dummy data (random integers as token ids)
    dummy_inputs = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    # Initialize long-term memory with zeros
    prev_memory = torch.zeros(batch_size, memory_dim).to(device)
    
    # Dummy values to simulate the GRPO advantage and log probabilities:
    # In a real setup, these would be computed from the model’s output probability distributions
    dummy_advantages = torch.rand(batch_size, device=device)
    dummy_old_log_probs = torch.log(torch.rand(batch_size, device=device) + 1e-8)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass: get logits and updated memory
        logits, new_memory = model(dummy_inputs, prev_memory)
        
        # For simplicity, we assume a language modeling task with a dummy target:
        # (e.g. predicting the first token, as a placeholder)
        targets = dummy_inputs[:, 0]
        ce_loss = F.cross_entropy(logits, targets)
        
        # Simulate new log probabilities (in practice, compute using the model's probability distribution)
        dummy_new_log_probs = torch.log(torch.rand(batch_size, device=device) + 1e-8)
        # Compute GRPO policy gradient loss
        pg_loss = grpo_loss(dummy_old_log_probs, dummy_new_log_probs, dummy_advantages)
        
        total_loss = ce_loss + pg_loss
        total_loss.backward()
        optimizer.step()
        
        # Update previous memory state
        prev_memory = new_memory.detach()
        
        print(f"Epoch {epoch+1}, Total Loss: {total_loss.item():.4f}, CE Loss: {ce_loss.item():.4f}, PG Loss: {pg_loss.item():.4f}")

if __name__ == "__main__":
    train_titans_model()















import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

# --------------------------------------------------
# Custom GRPO Trainer using TRL's PPOTrainer as a Base
# --------------------------------------------------
class GRPOTrainer(PPOTrainer):
    def compute_loss(self, query, response, rewards, **kwargs):
        """
        Custom loss computation using a GRPO-inspired update rule.
        This function assumes:
          - query: the prompt text (str)
          - response: the model's generated text (str)
          - rewards: a torch.Tensor of shape (B,) with advantage estimates
        The loss is computed by:
          1. Concatenating query and response.
          2. Tokenizing and computing log probabilities (summed over tokens)
             from both the current (new) policy and a reference (old) policy.
          3. Computing the ratio r = exp(new_log_prob - ref_log_prob) and clipping it.
          4. Forming the policy gradient loss and adding a (dummy) KL penalty.
        """
        # Concatenate query and response text
        full_text = query + response
        batch = self.tokenizer([full_text], return_tensors="pt", padding=True)
        batch = {k: v.to(self.model.device) for k, v in batch.items()}

        # Compute log probabilities from the reference (old) policy
        with torch.no_grad():
            ref_logits = self.ref_model(**batch).logits  # shape: (B, T, V)
            B, T, V = ref_logits.shape
            ref_logits_flat = ref_logits.view(-1, V)
            ref_labels_flat = batch["input_ids"].view(-1)
            ref_loss = F.cross_entropy(ref_logits_flat, ref_labels_flat, reduction="none")
            ref_log_probs = -ref_loss.view(B, T).sum(dim=1)  # shape: (B,)

        # Compute log probabilities from the current (new) policy
        new_logits = self.model(**batch).logits  # shape: (B, T, V)
        B, T, V = new_logits.shape
        new_logits_flat = new_logits.view(-1, V)
        labels_flat = batch["input_ids"].view(-1)
        new_loss = F.cross_entropy(new_logits_flat, labels_flat, reduction="none")
        new_log_probs = -new_loss.view(B, T).sum(dim=1)  # shape: (B,)

        # Compute probability ratio and clip it
        ratio = torch.exp(new_log_probs - ref_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.cliprange, 1 + self.config.cliprange)
        advantages = rewards  # assuming rewards are precomputed advantage estimates
        pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Dummy KL penalty (in practice, compute the actual KL divergence)
        kl_loss = (new_log_probs - ref_log_probs).mean()
        total_loss = pg_loss + self.config.kl_coef * kl_loss

        return total_loss

# --------------------------------------------------
# Main Fine-Tuning Script Using TRL and GRPOTrainer
# --------------------------------------------------
def main():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Using TRL's helper to add a value head (required for RL training)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    
    # Define PPO configuration (GRPO uses similar hyperparameters)
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=5e-5,
        log_with="wandb",        # or "tensorboard" or None
        batch_size=2,
        mini_batch_size=1,
        update_epochs=4,
        cliprange=0.15,
        kl_coef=0.0005,
    )
    
    # Instantiate our GRPOTrainer.
    # In this example, we use the same model as both the current and reference policy.
    grpo_trainer = GRPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=model,   # In practice, use a frozen copy from before the update.
        tokenizer=tokenizer,
    )
    
    # Dummy training data: a list of queries and corresponding generated responses.
    queries = [
        "The quick brown fox jumps over the lazy dog. ",
        "Deep learning models require vast amounts of data. "
    ]
    responses = [
        "It is a well-known pangram used to test fonts and keyboards.",
        "They are trained on millions of examples and optimized via gradient descent."
    ]
    
    # Dummy rewards (advantage estimates) for each sample.
    dummy_rewards = torch.tensor([1.0, 0.8]).to(model.device)
    
    # Run GRPO fine-tuning steps
    for query, response, reward in zip(queries, responses, dummy_rewards):
        stats = grpo_trainer.step(query, response, reward)
        print("GRPO update stats:", stats)

if __name__ == "__main__":
    main()
