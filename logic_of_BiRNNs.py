import torch
import torch.nn as nn

# Define toy vocabulary with 5 words (indexed from 0 to 4)
vocab_size = 5
embedding_dim = 4  # Each word will be represented by a 4-dimensional vector
hidden_dim = 3  # RNN hidden state size
sequence_length = 3  # Each sentence has 3 words

# Sample input sentence (word indices)
input_sentence = torch.tensor([[0, 1, 2]])  # A batch with 1 sentence of 3 words

embedding = nn.Embedding(vocab_size, embedding_dim)
embedded_sentence = embedding(input_sentence)

print("Word Embeddings Shape:", embedded_sentence.shape)  # Should be (1, 3, 4)

# Define forward and backward RNN weights manually
W_x_forward = nn.Linear(embedding_dim, hidden_dim, bias=False)  # Transform input
W_h_forward = nn.Linear(hidden_dim, hidden_dim, bias=True)  # Hidden state transition

W_x_backward = nn.Linear(embedding_dim, hidden_dim, bias=False)
W_h_backward = nn.Linear(hidden_dim, hidden_dim, bias=True)

# Activation function
tanh = nn.Tanh()

# Initialize hidden states (batch_size, hidden_dim)
h_t_forward = torch.zeros(1, hidden_dim)  # Initial hidden state forward
h_t_backward = torch.zeros(1, hidden_dim)  # Initial hidden state backward

# Lists to store output hidden states
hidden_states_forward = []
hidden_states_backward = []

# Forward Pass (Left to Right)
for t in range(sequence_length):
    x_t = embedded_sentence[:, t, :]  # Extract word embedding at time step t
    h_t_forward = tanh(W_x_forward(x_t) + W_h_forward(h_t_forward))
    hidden_states_forward.append(h_t_forward)

# Backward Pass (Right to Left)
for t in reversed(range(sequence_length)):
    x_t = embedded_sentence[:, t, :]
    h_t_backward = tanh(W_x_backward(x_t) + W_h_backward(h_t_backward))
    hidden_states_backward.append(h_t_backward)

# Reverse the backward hidden states to match order
hidden_states_backward.reverse()

# Concatenate forward and backward hidden states
hidden_states = [torch.cat((h_f, h_b), dim=1) for h_f, h_b in zip(hidden_states_forward, hidden_states_backward)]
hidden_states = torch.stack(hidden_states, dim=1)  # Shape: (batch_size, seq_length, hidden_dim * 2)

print("Final Hidden States Shape:", hidden_states.shape)  # Should be (1, 3, 6)

#==================================
#COMPARISION WITH BiRNNs
#=================================

# Built-in bidirectional RNN
rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
output, hidden = rnn(embedded_sentence)

print("PyTorch RNN Output Shape:", output.shape)  # (1, 3, 6) (bidirectional)
print("PyTorch Hidden Shape:", hidden.shape)  # (2, 1, 3) (2 directions, batch, hidden_dim)

