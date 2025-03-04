import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------
# 1️⃣ Sample NER Dataset (Word-Level Labels)
# ---------------------------------------
sentences = [
    ["Rahul", "lives", "in", "New", "Delhi", "."],
    ["Apple", "is", "a", "big", "company", "."],
    ["Jamsetji", "Tata", "founded", "IISc", "."],
]

labels = [
    ["B-PER", "O", "O", "B-LOC", "I-LOC", "O"],
    ["B-ORG", "O", "O", "O", "O", "O"],
    ["B-PER", "I-PER", "O", "B-ORG", "O"],
]

# ---------------------------------------
# 2️⃣ Vocabulary & Label Encoding (with <UNK> token)
# ---------------------------------------
word_vocab = {word: i+1 for i, word in enumerate(set(word for sent in sentences for word in sent))}
word_vocab["<PAD>"] = 0  # Padding token
word_vocab["<UNK>"] = len(word_vocab)  # Unknown token

tag_vocab = {tag: i for i, tag in enumerate(set(tag for tags in labels for tag in tags))}
tag_vocab["<PAD>"] = len(tag_vocab)  # Padding label

# Reverse Mapping
idx_to_tag = {i: tag for tag, i in tag_vocab.items()}

# Convert words & labels to indices
def encode_sentence(sent, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in sent]  # Use <UNK> if word is missing

def encode_labels(tags, vocab):
    return [vocab[tag] for tag in tags]

X = [encode_sentence(sent, word_vocab) for sent in sentences]
y = [encode_labels(tags, tag_vocab) for tags in labels]

# ---------------------------------------
# 3️⃣ Pad Sequences
# ---------------------------------------
max_len = max(len(sent) for sent in X)

def pad_sequence(seq, max_len, pad_value=0):
    return seq + [pad_value] * (max_len - len(seq))

X_padded = [pad_sequence(sent, max_len, word_vocab["<PAD>"]) for sent in X]
y_padded = [pad_sequence(tags, max_len, tag_vocab["<PAD>"]) for tags in y]

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_padded, dtype=torch.long)
y_tensor = torch.tensor(y_padded, dtype=torch.long)

# ---------------------------------------
# 4️⃣ Define NER Dataset & DataLoader
# ---------------------------------------
class NERDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = NERDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ---------------------------------------
# 5️⃣ Define NER Model (Simple RNN)
# ---------------------------------------
class NERRNN(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim=32, hidden_dim=64):
        super(NERRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tag_size)
    
    def forward(self, x):
        x = self.embedding(x)  # Word embeddings
        x, _ = self.rnn(x)     # RNN output
        x = self.fc(x)         # Linear layer to get predictions
        return x

# Initialize Model
vocab_size = len(word_vocab)
tag_size = len(tag_vocab)
model = NERRNN(vocab_size, tag_size)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tag_vocab["<PAD>"])  # Ignore padding labels
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---------------------------------------
# 6️⃣ Train the NER Model
# ---------------------------------------
num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0
    for words, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(words)  # Forward pass (batch, seq_len, tag_size)
        outputs = outputs.view(-1, tag_size)  # Flatten for loss computation
        labels = labels.view(-1)  # Flatten labels
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss:.4f}")

# ---------------------------------------
# 7️⃣ Test on a Sample Sentence (with <UNK> Handling)
# ---------------------------------------
def predict_sentence(sentence):
    encoded = encode_sentence(sentence, word_vocab)
    padded = pad_sequence(encoded, max_len, word_vocab["<PAD>"])
    tensor_input = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        output = model(tensor_input)
    
    predicted_tags = torch.argmax(output, dim=-1).squeeze(0).tolist()
    return [idx_to_tag[idx] for idx in predicted_tags[:len(sentence)]]  # Remove padding

# Example Test
test_sentence = ["Jamsetji", "Tata", "started", "IISc", "."]
predicted_labels = predict_sentence(test_sentence)

print("\nPredictions:")
for word, label in zip(test_sentence, predicted_labels):
    print(f"{word}: {label}")

