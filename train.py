# train.py

import gensim.downloader as api
import numpy as np

from model import SGNSModel
from utils import (
    generate_skip_grams,
    build_vocab,
    get_negative_distribution,
    most_similar
)

# -----------------------
# 1. Load dataset
# -----------------------

print("Loading text8 dataset...")
corpus = api.load("text8")

tokens = [word for sentence in corpus for word in sentence]

# limit dataset for demonstration
tokens = tokens[:1000000]

print(f"Number of tokens: {len(tokens)}")


# -----------------------
# 2. Build vocabulary
# -----------------------

vocab, id_to_word, word_counts = build_vocab(tokens)
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")

token_ids = np.array([vocab[word] for word in tokens])


# -----------------------
# 3. Generate skip-grams
# -----------------------

window_size = 2

centers, contexts = generate_skip_grams(token_ids, window_size)

print(f"Generated {len(centers)} center-context pairs")

num_pairs = len(centers)


# -----------------------
# 4. Negative sampling distribution
# -----------------------

noise_dist = get_negative_distribution(word_counts, id_to_word)


# -----------------------
# 5. Training parameters
# -----------------------

embed_dim = 50
learning_rate = 0.05
num_epochs = 3
k = 3                 # negative samples
batch_size = 512


# -----------------------
# 6. Initialize model
# -----------------------

model = SGNSModel(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    lr=learning_rate
)


# -----------------------
# 7. Training loop
# -----------------------

print("\nStarting training...\n")

for epoch in range(num_epochs):

    total_loss = 0
    num_batches = 0

    # shuffle data each epoch
    indices = np.random.permutation(num_pairs)

    centers_shuffled = centers[indices]
    contexts_shuffled = contexts[indices]

    for i in range(0, num_pairs, batch_size):

        batch_centers = centers_shuffled[i:i+batch_size]
        batch_contexts = contexts_shuffled[i:i+batch_size]

        B = len(batch_centers)

        # vectorized negative sampling
        neg_samples = np.random.choice(
            vocab_size,
            size=(B, k),
            p=noise_dist
        )

        # training step
        loss = model.train_step_batch(
            batch_centers,
            batch_contexts,
            neg_samples
        )

        total_loss += loss
        num_batches += 1

        if num_batches % 100 == 0:
            print(
                f"Epoch {epoch+1} | Batch {num_batches} | Avg Loss: {total_loss/num_batches:.4f}"
            )

    print(
        f"\nEpoch {epoch+1}/{num_epochs} finished | Avg Loss: {total_loss/num_batches:.4f}\n"
    )


# -----------------------
# 8. Evaluate embeddings
# -----------------------

print("\nMost similar words to 'king':")
print(most_similar("king", model.W, vocab, id_to_word, top_n=5))

print("\nMost similar words to 'queen':")
print(most_similar("queen", model.W, vocab, id_to_word, top_n=5))

print("\nMost similar words to 'man':")
print(most_similar("man", model.W, vocab, id_to_word, top_n=5))

print("\nMost similar words to 'woman':")
print(most_similar("woman", model.W, vocab, id_to_word, top_n=5))