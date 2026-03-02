# -----------------------
# 0. Imports
# -----------------------
import gensim.downloader as api
import numpy as np
from collections import Counter
import re

# -----------------------
# 1. Load and preprocess text8
# -----------------------
print("Loading text8 dataset...")
corpus = api.load("text8")  # iterable of lists of tokens

# flatten into a single list of tokens
tokens = [word for sentence in corpus for word in sentence]
tokens = tokens[:1_000_000]
print(f"Number of tokens: {len(tokens)}")

# -----------------------
# 2. Build vocabulary
# -----------------------
word_counts = Counter(tokens)
vocab = {word: i for i, (word, _) in enumerate(word_counts.items())}
id_to_word = {i: word for word, i in vocab.items()}
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Map tokens to IDs
token_ids = np.array([vocab[word] for word in tokens])

# -----------------------
# 3. Generate skip-gram pairs
# -----------------------
def generate_skip_grams(token_ids, window_size):
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

window_size = 2
centers, contexts = generate_skip_grams(token_ids, window_size)
print(f"Generated {len(centers)} center-context pairs")

# -----------------------
# 4. Negative sampling distribution
# -----------------------
word_freq = np.array([word_counts[id_to_word[i]] for i in range(vocab_size)])
noise_dist = word_freq ** 0.75
noise_dist = noise_dist / noise_dist.sum()

# -----------------------
# 5. SGNS Model class
# -----------------------
class SGNSModel:
    def __init__(self, vocab_size, embed_dim=50, lr=0.05):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = lr
        self.W = np.random.randn(vocab_size, embed_dim) * 0.01  # center embeddings
        self.U = np.random.randn(vocab_size, embed_dim) * 0.01  # context embeddings

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train_step(self, center, context, neg_samples):
        v_c = self.W[center]
        u_o = self.U[context]
        u_neg = self.U[neg_samples]

        # ---------- Forward ----------
        score_pos = np.dot(v_c, u_o)
        sig_pos = self.sigmoid(score_pos)

        score_neg = np.dot(u_neg, v_c)
        sig_neg = self.sigmoid(-score_neg)

        # ---------- Loss ----------
        loss = -np.log(sig_pos + 1e-10) - np.sum(np.log(sig_neg + 1e-10))

        # ---------- Gradients ----------
        grad_v = (sig_pos - 1) * u_o + np.sum((1 - sig_neg)[:, np.newaxis] * u_neg, axis=0)
        grad_uo = (sig_pos - 1) * v_c
        grad_un = (1 - sig_neg)[:, np.newaxis] * v_c[np.newaxis, :]

        # ---------- Update ----------
        self.W[center] -= self.lr * grad_v
        self.U[context] -= self.lr * grad_uo
        self.U[neg_samples] -= self.lr * grad_un

        return loss

# -----------------------
# 6. Training execution
# -----------------------
embed_dim = 50
lr = 0.05
num_epochs = 3  # small number for demonstration
k = 3  # number of negative samples

model = SGNSModel(vocab_size=vocab_size, embed_dim=embed_dim, lr=lr)

for epoch in range(num_epochs):
    total_loss = 0
    for i in range(len(centers)):
        center = centers[i]
        context = contexts[i]

        # sample k negatives
        neg_samples = []
        while len(neg_samples) < k:
            neg_word = np.random.choice(vocab_size, p=noise_dist)
            if neg_word != context:
                neg_samples.append(neg_word)
        neg_samples = np.array(neg_samples)

        # train step
        loss = model.train_step(center, context, neg_samples)
        total_loss += loss

        # optional: break early for demonstration purposes
        if i % 100000 == 0 and i > 0:
            print(f"Processed {i} pairs, avg loss: {total_loss/(i+1):.4f}")

    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {total_loss/len(centers):.4f}")

# -----------------------
# 7. Utility functions
# -----------------------
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def most_similar(word, W, vocab, top_n=5):
    if word not in vocab:
        return []
    idx = vocab[word]
    vec = W[idx]
    sims = []
    for i, w_vec in enumerate(W):
        sims.append(cosine_similarity(vec, w_vec))
    top_idx = np.argsort(sims)[::-1][1:top_n+1]  # exclude the word itself
    return [(id_to_word[i], sims[i]) for i in top_idx]

# -----------------------
# 8. Example usage
# -----------------------
print("\nMost similar words to 'king':")
print(most_similar("king", model.W, vocab, top_n=5))

print("\nMost similar words to 'queen':")
print(most_similar("queen", model.W, vocab, top_n=5))