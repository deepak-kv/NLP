import re
import random
import numpy as np
import torch
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
    def __init__(self, tokenizer):
        # Reserve index 0 for <UNK> and index 1 for [MASK]
        self.word2idx = {'<UNK>': 0, '[MASK]': 1}  # Initialize with special tokens
        self.idx2word = {0: '<UNK>', 1: '[MASK]'}  # Reverse mapping
        self.next_idx = 2  # Start assigning IDs from 2
        self.tokenizer = tokenizer  # Link the tokenizer to the vocabulary

    def add_word(self, word):
        """
        Adds a word to the vocabulary if it doesn't already exist.
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.next_idx
            self.idx2word[self.next_idx] = word
            self.next_idx += 1

    def build_vocab(self, sentences):
        """
        Builds the vocabulary from a list of sentences.
        """
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)  # Use the linked tokenizer
            for token in tokens:
                self.add_word(token)

    def get_token_ids(self, sentences, seq_length, masking=False, mask_prob=0.15):
        """
        Converts a list of sentences into token IDs using the vocabulary.
        Pads or truncates each sentence to ensure a fixed sequence length.
        If masking is True, randomly masks tokens according to mask_prob.
        Returns:
            - masked_token_ids: Token IDs with some tokens replaced by [MASK].
            - labels: Original token IDs for masked positions, -100 for non-masked positions.
        """
        token_ids = []
        labels = []
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)  # Use the linked tokenizer
            ids = [self.word2idx.get(token, 0) for token in tokens]  # Use 0 for unknown tokens

            # Apply masking if enabled
            if masking:
                masked_ids, label_ids = self._apply_masking(ids, mask_prob)
            else:
                masked_ids, label_ids = ids, [-100] * len(ids)

            # Pad or truncate the sequence to the specified seq_length
            if len(masked_ids) < seq_length:
                masked_ids += [0] * (seq_length - len(masked_ids))  # Pad with 0 (<UNK>)
                label_ids += [-100] * (seq_length - len(label_ids))  # Pad labels with -100
            else:
                masked_ids = masked_ids[:seq_length]  # Truncate to seq_length
                label_ids = label_ids[:seq_length]  # Truncate labels

            token_ids.append(masked_ids)
            labels.append(label_ids)

        return np.array(token_ids), np.array(labels)

    def _apply_masking(self, token_ids, mask_prob):
        """
        Randomly masks tokens in the input sequence according to mask_prob.
        Replaces masked tokens with:
        - [MASK] token (80% of the time),
        - A random token (10% of the time),
        - The original token (10% of the time).
        Returns:
            - masked_tokens: Token IDs with some tokens replaced.
            - labels: Original token IDs for masked positions, -100 for non-masked positions.
        """
        masked_tokens = []
        labels = []
        for token in token_ids:
            if random.random() < mask_prob:  # Mask this token
                rand = random.random()
                if rand < 0.8:  # 80% chance: replace with [MASK]
                    masked_tokens.append(self.word2idx['[MASK]'])
                elif rand < 0.9:  # 10% chance: replace with a random token
                    masked_tokens.append(random.choice(list(self.word2idx.values())))
                else:  # 10% chance: keep the original token
                    masked_tokens.append(token)
                labels.append(token)  # Save original token for loss computation
            else:  # Do not mask this token
                masked_tokens.append(token)
                labels.append(-100)  # Ignore in loss

        return masked_tokens, labels

    def convert_ids_to_tokens(self, ids):
        """
        Converts a list of token IDs to their corresponding tokens.
        Args:
            ids: List of token IDs.
        Returns:
            List of tokens.
        """
        return [self.idx2word.get(idx, '<UNK>') for idx in ids]

# Example usage (for testing the module)
if __name__ == "__main__":
    sentences = [
        'Everyone else in the building was outside, frightened and confused',
        ' They were using the screens and lights on their mobile phones to see better',
        ' Several people got in their cars and turned on the lights',
        ' They drove to the entrance to make a small area of light for everybody to stand together',
        'The street lights turned on, but most people were still afraid',
        ' '
    ]

    #tokenizer = CustomTokenizer()
    #vocab = Vocabulary(tokenizer)  # Pass the tokenizer to the Vocabulary

    # Build the vocabulary
    #vocab.build_vocab(sentences)

    # Define the desired
