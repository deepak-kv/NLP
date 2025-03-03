import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from itertools import product
import re

# ---------------------------------------
# 1️⃣ Custom Text Corpus
# ---------------------------------------
custom_corpus = """
Deep learning models require large amounts of data for training.
Natural Language Processing is an application of deep learning.
Word embeddings capture the semantic meaning of words in a text.
GloVe is a method to learn word embeddings using co-occurrence.
"""

# Tokenization
words = re.findall(r'\b\w+\b', custom_corpus.lower())
vocab = list(set(words))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

vocab_size = len(vocab)
window_size = 2  # Context window size

# ---------------------------------------
# 2️⃣ Build Co-occurrence Matrix
# ---------------------------------------
co_occurrence = np.zeros((vocab_size, vocab_size))

for i, word in enumerate(words):
    word_idx = word_to_idx[word]
    context_indices = list(range(max(0, i - window_size), min(len(words), i + window_size + 1)))
    context_indices.remove(i)  # Remove center word
    for j in context_indices:
        context_idx = word_to_idx[words[j]]
        co_occurrence[word_idx, context_idx] += 1  # Count co-occurrences

# Convert to PyTorch Tensor
X = torch.tensor(co_occurrence, dtype=torch.float)

# ---------------------------------------
# 3️⃣ Define GloVe Model
# ---------------------------------------
class GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloVe, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.target_bias = nn.Parameter(torch.zeros(vocab_size))
        self.context_bias = nn.Parameter(torch.zeros(vocab_size))
        
        # nn.init.xavier_uniform_(self.target_embeddings.weight)
        # nn.init.xavier_uniform_(self.context_embeddings.weight)

    def forward(self, target, context):
        v_target = self.target_embeddings(target)  # Target word embedding
        v_context = self.context_embeddings(context)  # Context word embedding
        bias_target = self.target_bias[target]  # Bias for target
        bias_context = self.context_bias[context]  # Bias for context

        dot_product = (v_target * v_context).sum(dim=1)  # Compute dot product
        loss = (dot_product + bias_target + bias_context - torch.log(X[target, context] + 1)).pow(2)  # GloVe Loss
        return loss.mean()

# ---------------------------------------
# 4️⃣ Train the GloVe Model
# ---------------------------------------
embedding_dim = 5
model = GloVe(vocab_size, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    total_loss = 0
    for i, j in product(range(vocab_size), repeat=2):
        if X[i, j] > 0:  # Only train on non-zero co-occurrences
            
            optimizer.zero_grad()
            loss = model(torch.tensor([i]), torch.tensor([j]))  # Forward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss:.4f}")

# ---------------------------------------
# 5️⃣ Extract & Use Word Embeddings
# ---------------------------------------
word_embeddings = model.target_embeddings.weight.data.numpy()

def get_embedding(word):
    return word_embeddings[word_to_idx[word]]

# Example: Get vector for "deep"
deep_vector = get_embedding("deep")
print("\nWord Vector for 'deep':", deep_vector)

