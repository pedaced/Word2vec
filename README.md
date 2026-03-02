# Word2Vec Skip-Gram with Negative Sampling (SGNS) Implementation

## Overview

This project implements a **Skip-Gram Word2Vec model with Negative Sampling (SGNS)** in **pure NumPy**. 

We chose SGNS because it is **the most common and efficient approach** for learning word embeddings from large corpora. SGNS captures semantic relationships between words by training a model to **predict context words given a center word**, while using **negative sampling** to efficiently approximate the softmax over a large vocabulary.

---

## Project Pipeline

The code is organized into several key steps, which together form the full SGNS training pipeline:

### 1. Load and Preprocess Data

- The **Gensim `text8` dataset** is used as the corpus. This is a clean, preprocessed text from Wikipedia.
- Tokens are extracted and mapped to integer IDs.
- Vocabulary and token counts are built to track word frequencies.

### 2. Generate Skip-Gram Pairs

- For each center word, a **context window** is defined (e.g., 1–2 words on each side).
- **Center-context pairs** are generated for all tokens in the corpus.
- This forms the positive training examples for the SGNS model.

### 3. Build Negative Sampling Distribution

- Word frequencies are raised to the **0.75 power** (following Mikolov et al.).
- This creates a **noise distribution** from which negative samples are drawn.
- Negative sampling helps the model distinguish true context words from random words efficiently.

### 4. Define the SGNS Model

- The `SGNSModel` class contains:
  - Two embedding matrices: one for **center words** (`W`) and one for **context words** (`U`).
  - **Forward pass** to compute probabilities using the sigmoid of dot products.
  - **Loss computation** using the SGNS loss function.
  - **Backward pass** to compute gradients for the embeddings.
  - **Parameter update** using stochastic gradient descent (SGD).

### 5. Training Loop

- For each epoch:
  - Loop through all center-context pairs.
  - Sample `k` negative words for each positive pair.
  - Perform a **train step** to update embeddings.
  - Track the **average loss** to monitor training progress.
- After training, the `W` matrix contains the learned **word embeddings**.

### 6. Utility Functions

- **Cosine similarity**: measure similarity between two embeddings.
- **Most similar words**: find top-N words closest to a given word in embedding space.
- Optional: word analogy tasks can be performed using vector arithmetic.

---

## Notes

- Training on the **full text8 dataset** is computationally heavy on CPU. For experiments, a **subset of tokens** can be used.
- Smaller datasets and window sizes may result in embeddings that do not capture meaningful semantics.
- Increasing **corpus size, window size, embedding dimensions, negative samples, and epochs** improves embedding quality.

---

## Example Usage

```python
# Find words most similar to 'king'
most_similar("king", model.W, vocab, top_n=5)

# Find words most similar to 'queen'
most_similar("queen", model.W, vocab, top_n=5)