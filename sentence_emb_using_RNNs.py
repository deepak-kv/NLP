import torch
import torch.nn as nn

class SentenceEncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentenceEncoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Word Embeddings
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)  # Convert words to embeddings
        output, hidden = self.rnn(embedded)  
        return hidden.squeeze(0)  # Final hidden state = sentence embedding

# Example Usage
vocab_size = 5000
embedding_dim = 300
hidden_dim = 128
model = SentenceEncoderRNN(vocab_size, embedding_dim, hidden_dim)

sample_sentence = torch.randint(0, vocab_size, (1, 5))  # Example sentence with 5 words
sentence_embedding = model(sample_sentence)
print(sentence_embedding.shape)  # Output: torch.Size([128])

