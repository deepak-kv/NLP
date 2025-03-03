import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from collections import Counter
import re
# ---------------------------------------
#  Custom Text Corpus
# ---------------------------------------
custom_corpus = """
Deep learning models require large amounts of data for training.
Natural Language Processing is an application of deep learning.
Word embeddings capture the semantic meaning of words in a text.
Skip-gram is a method to learn word embeddings using a neural network.
"""

# ---------------------------------------
#  Tokenization & Vocabulary Building
# ---------------------------------------
#words = custom_corpus.lower().split()  # Simple tokenization
words = re.findall(r'\b\w+\b', custom_corpus.lower())

word_counts = Counter(words)  # Count word frequencies
vocab = list(word_counts.keys())  # Unique words
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

vocab_size = len(vocab)
window_size = 2  # Context window size

# ---------------------------------------
#  Generate Word Pairs for Skip-gram
# ---------------------------------------
def generate_skipgram_data(words, window_size):
    pairs = []
    for i, word in enumerate(words):
        target_word = word_to_idx[word]  # Center word
        context_start = max(0, i - window_size)
        context_end = min(len(words), i + window_size + 1)

        for j in range(context_start, context_end):
            if i != j:  # Exclude the center word itself
                context_word = word_to_idx[words[j]]
                pairs.append((target_word, context_word))

    return pairs

training_data = generate_skipgram_data(words, window_size)

# ---------------------------------------
#  Define Custom Dataset Class
# ---------------------------------------
class SkipGramDataset(Dataset):
    def __init__(self, word_pairs):
        self.word_pairs = word_pairs  # (center, context) pairs
    
    def __len__(self):
        return len(self.word_pairs)
    
    def __getitem__(self, idx):
        center, context = self.word_pairs[idx]  # Unpack tuple
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)  # Return as two separate tensors


# Create Dataset and DataLoader
dataset = SkipGramDataset(training_data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ---------------------------------------
#  Define Skip-gram Model
# ---------------------------------------
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # Word embeddings
        self.output_layer = nn.Linear(embedding_dim, vocab_size, bias=False)  # Output layer
    
    def forward(self, center_word):
        embed = self.embeddings(center_word)  # (batch_size, embedding_dim)
        logits = self.output_layer(embed)  # (batch_size, vocab_size)
        return logits

# Initialize Model
embedding_dim = 50
model = SkipGram(vocab_size, embedding_dim)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---------------------------------------
#  Train the Skip-gram Model
# ---------------------------------------
num_epochs = 1000
for epoch in range(num_epochs):
    total_loss = 0
    for center, context in dataloader:
        optimizer.zero_grad()
        outputs = model(center)  # Forward pass
        loss = criterion(outputs, context)  # Compute loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss:.4f}")

# ---------------------------------------
#  Extract & Use Word Embeddings
# ---------------------------------------
word_embeddings = model.embeddings.weight.data.numpy()

def get_embedding(word):
    return word_embeddings[word_to_idx[word]]

# Example: Get vector for "deep"
deep_vector = get_embedding("deep")
print("\nWord Vector for 'deep':", deep_vector)





# ---------------------------
# Visualization
# --------------------------
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# Reduce to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
tsne_embeddings = tsne.fit_transform(word_embeddings)

# Plot
plt.figure(figsize=(10, 6))
for i, word in enumerate(word_to_idx):
    x, y = tsne_embeddings[i]
    plt.scatter(x, y)
    plt.text(x, y, word, fontsize=12)
plt.title("Word Embeddings Visualization using t-SNE")
plt.show()

# Save plot as PNG
plt.savefig("word_embeddings_tsne.png", dpi=300)  # Save with high resolution
plt.close() 
