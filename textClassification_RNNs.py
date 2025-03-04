import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)





# Sample Dataset (Sentences & Labels)
train_data = [
    ("I love this movie, it's amazing!", 1),   # Positive
    ("This film was a complete waste of time.", 0),  # Negative
    ("Fantastic performance by the actors.", 1),  # Positive
    ("I hate this movie, it was boring.", 0),  # Negative
    ("One of the best movies I have ever seen!", 1),  # Positive
    ("The plot was terrible and acting was worse.", 0)  # Negative
]



# Vocabulary and encoding
word_to_idx = {"<PAD>": 0, "I": 1, "love": 2, "this": 3, "movie": 4, "it's": 5, 
               "amazing!": 6, "film": 7, "was": 8, "a": 9, "complete": 10, "waste": 11, 
               "of": 12, "time.": 13, "fantastic": 14, "performance": 15, "by": 16, 
               "the": 17, "actors.": 18, "hate": 19, "boring.": 20, "one": 21, "best": 22, 
               "movies": 23, "have": 24, "ever": 25, "seen!": 26, "plot": 27, 
               "terrible": 28, "and": 29, "acting": 30, "worse.": 31}

vocab_size = len(word_to_idx)  # Total number of words
embedding_dim = 8  # Size of word embeddings
hidden_dim = 16  # Size of hidden state in RNN
num_classes = 2  # Positive or Negative

# Convert text to numerical sequences
def encode_sentence(sentence, word_to_idx, max_len=10):
    encoded = [word_to_idx.get(word, 0) for word in sentence.split()]
    encoded = encoded[:max_len] + [0] * (max_len - len(encoded))  # Padding
    return encoded

# Custom Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, data, word_to_idx):
        self.data = [(torch.tensor(encode_sentence(sent, word_to_idx)), torch.tensor(label)) for sent, label in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Define RNN Model for Sentiment Analysis
class RNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(RNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Convert word indices to embeddings
        _, h_n = self.rnn(x)  # Apply RNN
        out = self.fc(h_n.squeeze(0))  # Fully connected layer
        return out

# Create dataset and dataloader
train_dataset = SentimentDataset(train_data, word_to_idx)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Initialize Model
model = RNNTextClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train Model
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Test the Model with an Example
test_sentence = "I love this movie"
test_encoded = torch.tensor([encode_sentence(test_sentence, word_to_idx)])
test_output = model(test_encoded)
predicted_label = torch.argmax(test_output, dim=1).item()

print("\nTest Sentence:", test_sentence)
print("Predicted Sentiment:", "Positive" if predicted_label == 1 else "Negative")

