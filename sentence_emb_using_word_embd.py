# ---------------------------------------
# 6️⃣ Generate Sentence Embeddings
# ---------------------------------------
word_embeddings = model.embeddings.weight.data.numpy()

def get_word_embedding(word):
    return word_embeddings[word_to_idx[word]]

def get_sentence_embedding(sentence):
    words = sentence.lower().split()
    word_vectors = [get_word_embedding(word) for word in words if word in word_to_idx]
    if not word_vectors:
        return None  # Return None if no words match
    return torch.tensor(word_vectors).mean(dim=0)  # Averaging word embeddings

# Example Usage:
sentence = "deep learning models"
sentence_embedding = get_sentence_embedding(sentence)
print("\nSentence Embedding for:", sentence)
print(sentence_embedding)




# sentence embedding using tf-idf weights


# ---------------------------------------
# 3️⃣ Compute TF-IDF Weights for Words
# ---------------------------------------
vectorizer = TfidfVectorizer()
vectorizer.fit(custom_corpus)
tfidf_matrix = vectorizer.transform(custom_corpus)
word_tfidf = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

def get_weighted_sentence_embedding(sentence):
    words = sentence.lower().split()
    word_vectors = []
    weights = []

    for word in words:
        if word in word_to_idx:
            embedding = get_word_embedding(word)
            weight = word_tfidf.get(word, 1.0)  # Default weight 1.0 if not in TF-IDF
            word_vectors.append(embedding * weight)
            weights.append(weight)
    
    if not word_vectors:
        return None  # No valid words in the sentence
    
    sentence_embedding = np.sum(word_vectors, axis=0) / np.sum(weights)  # Weighted average
    return torch.tensor(sentence_embedding)


# ---------------------------------------
# 4️⃣ Example: Compute Weighted Sentence Embedding
# ---------------------------------------
sentence = "deep learning models"
weighted_embedding = get_weighted_sentence_embedding(sentence)

print("\nWeighted Sentence Embedding for:", sentence)
print(weighted_embedding)
