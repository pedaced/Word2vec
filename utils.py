# utils.py
import numpy as np
from collections import Counter

# -----------------------
# Preprocessing
# -----------------------
def generate_skip_grams(token_ids, window_size=2):
    centers = []
    contexts = []
    for idx, center_id in enumerate(token_ids):
        start = max(0, idx - window_size)
        end = min(len(token_ids), idx + window_size + 1)
        for context_idx in range(start, end):
            if context_idx != idx:
                centers.append(center_id)
                contexts.append(token_ids[context_idx])
    return np.array(centers), np.array(contexts)

def build_vocab(tokens):
    word_counts = Counter(tokens)
    vocab = {word: i for i, (word, _) in enumerate(word_counts.items())}
    id_to_word = {i: word for word, i in vocab.items()}
    return vocab, id_to_word, word_counts

def get_negative_distribution(word_counts, id_to_word):
    vocab_size = len(word_counts)
    word_freq = np.array([word_counts[id_to_word[i]] for i in range(vocab_size)])
    noise_dist = word_freq ** 0.75
    noise_dist /= noise_dist.sum()
    return noise_dist

# -----------------------
# Similarity utilities
# -----------------------
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def most_similar(word, W, vocab, id_to_word, top_n=5):
    if word not in vocab:
        return []
    idx = vocab[word]
    vec = W[idx]
    sims = []
    for i, w_vec in enumerate(W):
        sims.append(cosine_similarity(vec, w_vec))
    top_idx = np.argsort(sims)[::-1][1:top_n+1]  # exclude the word itself
    return [(id_to_word[i], sims[i]) for i in top_idx]