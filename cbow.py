import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
# ---------------------------------------
# 1️⃣ Custom Text Corpus
# ---------------------------------------
custom_corpus = """
Deep learning models require large amounts of data for training.
Natural Language Processing is an application of deep learning.
Word embeddings capture the semantic meaning of words in a text.
CBOW is a method to learn word embeddings using a neural network.
"""

# ---------------------------------------
# 2️⃣ Tokenization & Vocabulary Building
# ---------------------------------------
words = re.findall(r'\b\w+\b', custom_corpus.lower())  # Simple tokenization
word_counts = Counter(words)  # Count word frequencies
vocab = list(word_counts.keys())  # Unique words
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

vocab_size = len(vocab)
window_size = 2  # Context window size

# ---------------------------------------
# 3️⃣ Generate Training Data for CBOW
# ---------------------------------------
def generate_cbow_data(words, window_size):
    pairs = []
    for i in range(window_size, len(words) - window_size):
        context_words = [word_to_idx[words[j]] for j in range(i - window_size, i + window_size + 1) if j != i]
        target_word = word_to_idx[words[i]]
        pairs.append((context_words, target_word))
    return pairs

training_data = generate_cbow_data(words, window_size)

# ---------------------------------------
# 4️⃣ Define CBOW Dataset Class
# ---------------------------------------
class CBOWDataset(Dataset):
    def __init__(self, word_pairs):
        self.word_pairs = word_pairs  # (context_words, target_word) pairs
    
    def __len__(self):
        return len(self.word_pairs)
    
    def __getitem__(self, idx):
        context, target = self.word_pairs[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# Create Dataset and DataLoader
dataset = CBOWDataset(training_data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ---------------------------------------
# 5️⃣ Define CBOW Model
# ---------------------------------------
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, context_words):
        embed = self.embeddings(context_words)  # (batch_size, context_size, embedding_dim)
        embed = embed.mean(dim=1)  # Averaging context embeddings
        logits = self.linear(embed)  # (batch_size, vocab_size)
        return logits

# Initialize Model
embedding_dim = 5
model = CBOW(vocab_size, embedding_dim)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---------------------------------------
# 6️⃣ Train the CBOW Model
# ---------------------------------------
num_epochs = 1000
for epoch in range(num_epochs):
    total_loss = 0
    for context, target in dataloader:
        optimizer.zero_grad()
        outputs = model(context)  # Forward pass
        loss = criterion(outputs, target)  # Compute loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss:.4f}")

# ---------------------------------------
# 7️⃣ Extract & Use Word Embeddings
# ---------------------------------------
word_embeddings = model.embeddings.weight.data.numpy()

def get_embedding(word):
    return word_embeddings[word_to_idx[word]]

# Example: Get vector for "deep"
deep_vector = get_embedding("deep")
print("\nWord Vector for 'deep':", deep_vector)

