

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
import random


#======================================================================
#                       Multi-Head-Attention
#======================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear layers for Q, K, V
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        batch_size, seq_length, hidden_dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # Shape: (batch, seq_length, hidden_dim * 3)
        qkv = qkv.view(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # Split into three parts

        # Transpose for multi-head attention: (batch, seq_length, num_heads, head_dim) -> (batch, num_heads, seq_length, head_dim)
        q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Shape: (batch, num_heads, seq_length, seq_length)

        # Apply mask (if provided)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]  # Reshape to (batch, 1, 1, seq_length)
            attention_mask = attention_mask.expand(-1, self.num_heads, seq_length, seq_length)  # Expand to match scores shape
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Softmax and weighted sum
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)  # Shape: (batch, num_heads, seq_length, head_dim)

        # Reshape and pass through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_dim)  # Reshape back
        return self.fc_out(output)




#======================================================================
#                       Trnsformer Encoder Block
#======================================================================

class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, attention_mask=None):
        # Multi-head attention + residual connection
        attn_output = self.attention(x, attention_mask)
        x = self.norm1(x + attn_output)

        # Feed-forward network + residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


#======================================================================
#                       Trnsformer Encoder
#======================================================================

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, vocab_size, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        positions = torch.arange(0, seq_length, device=input_ids.device).unsqueeze(0)

        # Add word and position embeddings (segment embedding removed)
        x = self.embedding(input_ids) + self.position_embedding(positions)

        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, attention_mask)

        return self.norm(x)



#=====================================================================
#                   Mask Language Model (MLM)
#======================================================================



# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Small synthetic dataset
sentences = [
    "The cat sat on the mat.",
    "The sun is shining brightly.",
    "I love deep learning and NLP.",
    "Transformers have changed AI research.",
    "PyTorch makes deep learning easier."
]

# Tokenize sentences and apply masking
def mask_tokens(sentence, mask_prob=0.15):
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    masked_tokens = []
    labels = []
    for token in token_ids:
        if random.random() < mask_prob:
            masked_tokens.append(tokenizer.mask_token_id)  # Replace with [MASK]
            labels.append(token)  # Save original token for loss computation
        else:
            masked_tokens.append(token)
            labels.append(-100)  # Ignore in loss
    
    return masked_tokens, labels

# Prepare dataset
input_ids = []
labels = []
for sentence in sentences:
    masked_input, label = mask_tokens(sentence)
    input_ids.append(masked_input)
    labels.append(label)

# Pad sequences to the maximum length in the batch
max_len = max(len(masked) for masked in input_ids)
input_ids = [seq + [tokenizer.pad_token_id] * (max_len - len(seq)) for seq in input_ids]
labels = [seq + [-100] * (max_len - len(seq)) for seq in labels]

input_ids = torch.tensor(input_ids, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.long)


# Define simple MLM model (using a TransformerEncoder)
class SimpleMLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_heads=4, num_layers=2, max_length=128):
        super().__init__()
        self.encoder = TransformerEncoder(hidden_dim, num_heads, num_layers, vocab_size, max_length)
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Output layer to predict words
    
    def forward(self, input_ids, attention_mask=None):
        x = self.encoder(input_ids, attention_mask)
        return self.fc(x)  # Predict next token


# Initialize model
vocab_size = tokenizer.vocab_size
model = SimpleMLM(vocab_size)

# Training setup
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(input_ids)
    loss = criterion(output.view(-1, vocab_size), labels.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Test prediction (fill in masked words)
with torch.no_grad():
    test_output = model(input_ids)
    predicted_ids = test_output.argmax(dim=-1)
    predicted_tokens = [tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in predicted_ids]
    
    print("\nPredicted sentences:")
    for i, tokens in enumerate(predicted_tokens):
        print(" ".join(tokens))

