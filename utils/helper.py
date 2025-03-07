import numpy as np
from collections import defaultdict

class CustomTokenizer:
    def __init__(self):
        # Define a simple tokenizer that splits text into words and converts to lowercase
        self.pattern = r'\w+|\S'  # Matches words or single non-whitespace characters

    def tokenize(self, text):
        """
        Tokenizes the input text based on the defined pattern.
        """
        tokens = re.findall(self.pattern, text.lower())  # Convert to lowercase for consistency
        return tokens

class Vocabulary:
    def __init__(self):
        self.word2idx = defaultdict(lambda: len(self.word2idx))  # Automatically assign new IDs
        self.idx2word = {}

    def build_vocab(self, sentences):
        """
        Builds the vocabulary from a list of sentences.
        """
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            for token in tokens:
                _ = self.word2idx[token]  # Assign an ID to each token

        # Create a reverse mapping for decoding
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def get_token_ids(self, sentences, seq_length):
        """
        Converts a list of sentences into token IDs using the vocabulary.
        Pads or truncates each sentence to ensure a fixed sequence length.
        """
        token_ids = []
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            ids = [self.word2idx[token] for token in tokens]
            # Pad or truncate the sequence to the specified seq_length
            if len(ids) < seq_length:
                ids += [0] * (seq_length - len(ids))  # Pad with 0 (assuming 0 is the padding token)
            else:
                ids = ids[:seq_length]  # Truncate to seq_length
            token_ids.append(ids)
        return np.array(token_ids).T  # Transpose to get shape (seq_length, num_sentences)

# Input sentences
#sentences = [
#    'Everyone else in the building was outside, frightened and confused',
#    ' They were using the screens and lights on their mobile phones to see better',
#    ' Several people got in their cars and turned on the lights',
#    ' They drove to the entrance to make a small area of light for everybody to stand together',
#    'The street lights turned on, but most people were still afraid',
#    ' '
#]

'''
# Initialize tokenizer and vocabulary
tokenizer = CustomTokenizer()
vocab = Vocabulary()

# Build the vocabulary
vocab.build_vocab(sentences)

# Define the desired sequence length
seq_length = 15  # Example sequence length

# Get token IDs for the sentences in the shape (seq_length, num_sentences)
token_ids = vocab.get_token_ids(sentences, seq_length)
vocab_size = len(vocab.word2idx)
token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)

# Print the results
print("Vocabulary:")
print(vocab.word2idx)
print("\nToken IDs (shape: (seq_length, num_sentences)):")
print(token_ids)
'''
