import re
import numpy as np
from collections import Counter

# -----------------------
# 1. Raw text
# -----------------------
text = "I love cats and dogs. Cats are awesome!"

# -----------------------
# 2. Lowercase and clean
# -----------------------
text = text.lower()
tokens = re.findall(r'\b\w+\b', text)
print("Tokens:", tokens)

# -----------------------
# 3. Build vocabulary
# -----------------------
word_counts = Counter(tokens)
vocab = {word: i for i, (word, _) in enumerate(word_counts.items())}
id_to_word = {i: word for word, i in vocab.items()}
vocab_size = len(vocab)
print("Vocabulary:", vocab)
print("Vocabulary size:", vocab_size)

# -----------------------
# 4. Map tokens to IDs
# -----------------------
print(tokens)
token_ids = np.array([vocab[word] for word in tokens])
print("Token IDs:", token_ids)

# -----------------------
# 6. Generate skip-gram pairs
# -----------------------
def generate_skip_grams(tokens, token_ids, window_size):
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

window_size = 1
centers, contexts = generate_skip_grams(tokens, token_ids, window_size)
print("Positive pairs (center, context):", list(zip(centers, contexts)))

# -----------------------
# 7. Build negative sampling distribution
# -----------------------
word_freq = np.array([word_counts[id_to_word[i]] for i in range(vocab_size)])
noise_dist = word_freq ** 0.75
noise_dist = noise_dist / noise_dist.sum()

# -----------------------
# 8. Generate negative samples
# -----------------------
def generate_negative_samples(centers, contexts, vocab_size, noise_dist, k):
    neg_centers = []
    neg_contexts = []

    N = len(contexts)
    for i in range(N):
        center = centers[i]
        true_context = contexts[i]
        count = 0
        while count < k:
            neg_word = np.random.choice(vocab_size, p=noise_dist)
            if neg_word != true_context:
                neg_centers.append(center)
                neg_contexts.append(neg_word)
                count += 1

    return np.array(neg_centers), np.array(neg_contexts)

k = 3
neg_centers, neg_contexts = generate_negative_samples(centers, contexts, vocab_size, noise_dist, k)
print("\nNegative pairs (center, negative):", list(zip(neg_centers, neg_contexts)))

# -----------------------
# Now you have:
# - centers, contexts (positive pairs)
# - neg_centers, neg_contexts (negative samples)
# Ready for SGNS training
# -----------------------