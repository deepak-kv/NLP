import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
import random

import torch
import torch.nn as nn

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

# Define Transformer Encoder Layer
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, vocab_size, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_length = input_ids.shape
        positions = torch.arange(0, seq_length, device=input_ids.device).unsqueeze(0)
        
        x = self.embedding(input_ids) + self.position_embedding(positions)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        return self.norm(x[:, 0, :])  # Use CLS token representation






# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Small synthetic dataset for NSP
sentence_pairs = [
    ("The cat sat on the mat.", "It looked around cautiously.", 1),  # IsNext
    ("The sun is shining brightly.", "I love deep learning and NLP.", 0),  # NotNext
    ("Transformers have changed AI research.", "PyTorch makes deep learning easier.", 0),
    ("I went to the park today.", "The trees were full of birds.", 1)
]

# Tokenize sentences and prepare input

def encode_sentence_pair(sent1, sent2, label, max_length=20):
    tokens = tokenizer(sent1, sent2, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    return tokens['input_ids'].squeeze(0), tokens['token_type_ids'].squeeze(0), tokens['attention_mask'].squeeze(0), torch.tensor(label)

# Prepare dataset
input_ids, segment_ids, attention_masks, labels = zip(*[encode_sentence_pair(s1, s2, lbl) for s1, s2, lbl in sentence_pairs])
input_ids = torch.stack(input_ids)
segment_ids = torch.stack(segment_ids)
attention_masks = torch.stack(attention_masks)
labels = torch.stack(labels)

# Define Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, vocab_size, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.segment_embedding = nn.Embedding(2, hidden_dim)  # Segment IDs: 0 or 1
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, segment_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        positions = torch.arange(0, seq_length, device=input_ids.device).unsqueeze(0)
        
        # Add word, position, and segment embeddings
        x = self.embedding(input_ids) + self.position_embedding(positions) + self.segment_embedding(segment_ids)
        
        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        return self.norm(x)

# Define NSP Model
class NSPModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_heads=4, num_layers=2, max_length=128):
        super().__init__()
        self.encoder = TransformerEncoder(hidden_dim, num_heads, num_layers, vocab_size, max_length)
        self.fc = nn.Linear(hidden_dim, 2)  # Binary classification (IsNext or NotNext)
    
    def forward(self, input_ids, segment_ids, attention_mask):
        x = self.encoder(input_ids, segment_ids, attention_mask)
        cls_output = x[:, 0, :]  # Use CLS token embedding for classification
        return self.fc(cls_output)

# Initialize model
vocab_size = tokenizer.vocab_size
model = NSPModel(vocab_size)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(input_ids, segment_ids, attention_masks)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    print (f"input_ids, segment_ids, {segment_ids}")
    print (f" attention_masks : {attention_masks}")

# Test prediction
with torch.no_grad():
    test_output = model(input_ids, segment_ids, attention_masks)
    predicted_labels = torch.argmax(test_output, dim=1)
    print("\nPredicted NSP Labels:", predicted_labels.tolist())

