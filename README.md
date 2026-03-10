# Word2Vec (Skip-Gram with Negative Sampling) – Pure NumPy Implementation

This repository contains a **from-scratch implementation of Word2Vec** using **pure NumPy**.  
No deep learning frameworks (PyTorch, TensorFlow, etc.) are used.

The goal of this project is to implement the **core training loop of Word2Vec**, including:

- forward pass
- loss computation
- gradient derivation
- parameter updates

The model implemented is **Skip-Gram with Negative Sampling (SGNS)**, one of the most commonly used Word2Vec variants.

---

# 1. Overview

Word2Vec learns **dense vector representations of words** such that words appearing in similar contexts have similar embeddings.

For example, embeddings should capture relationships like:

```
king - man + woman ≈ queen
```

or semantic similarity:

```
king ≈ queen
car ≈ vehicle
```

This implementation closely follows the algorithm introduced by:

**Tomas Mikolov et al. (2013)**  
*Efficient Estimation of Word Representations in Vector Space*

---

# 2. Skip-Gram with Negative Sampling

The model predicts **context words given a center word**.

For a center word \(w_c\) and context word \(w_o\), the objective is:

\[
\log \sigma(u_o^T v_c) + \sum_{k=1}^{K} \log \sigma(-u_k^T v_c)
\]

Where:

- \(v_c\) = center word embedding
- \(u_o\) = context embedding
- \(u_k\) = embeddings of negative samples
- \(K\) = number of negative samples
- \(\sigma\) = sigmoid function

Negative samples are drawn from a **noise distribution proportional to word frequency^0.75**, as proposed in the original paper.

Two embedding matrices are learned:

```
W : center word embeddings   (vocab_size × embedding_dim)
U : context word embeddings  (vocab_size × embedding_dim)
```

Training updates both matrices using **stochastic gradient descent**.

---

# 3. Dataset

The model is trained on the **text8 dataset**, loaded using:

```
gensim.downloader
```

For faster training during experimentation, the corpus is truncated to:

```
1,000,000 tokens
```

This provides a good balance between:

- meaningful word statistics
- reasonable CPU training time

---

# 4. Training Configuration

| Parameter | Value |
|--------|------|
| Model | Skip-Gram |
| Window size | 2 |
| Embedding dimension | 50 |
| Negative samples | 3 |
| Learning rate | 0.05 |
| Epochs | 3 |

---

# 5. Project Structure

```
.
├── train_word2vec.py
└── README.md
```

---

# 6. Installation

Install the required dependencies:

```
pip install numpy gensim
```

---

# 7. Running the Training

Run the training script:

```
python train_word2vec.py
```

The script will:

1. Load the **text8 dataset**
2. Build the **vocabulary**
3. Generate **skip-gram training pairs**
4. Initialize embedding matrices
5. Train the model using **negative sampling**
6. Print training progress and loss

Example output:

```
Processed 100000 pairs, avg loss: 2.13
Processed 200000 pairs, avg loss: 2.05
Epoch 1/3 Avg Loss: 1.98
```

---

# 8. Evaluating the Learned Embeddings

After training, the quality of the embeddings can be evaluated using **cosine similarity**.

Two common intrinsic evaluation methods are implemented:

### 1. Nearest Neighbor Search

Find words with similar embeddings.

Example:

```
Most similar words to 'king':
queen
prince
monarch
throne
emperor
```

### 2. Word Analogies

Classic analogy test:

```
king - man + woman ≈ queen
```

---

# 9. Evaluation Script

The following utility functions can be used to evaluate embeddings.

```python
import numpy as np


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )


def most_similar(word, W, vocab, id_to_word, top_n=5):
    """
    Returns the top-N most similar words based on cosine similarity.
    """

    if word not in vocab:
        print(f"{word} not in vocabulary")
        return []

    idx = vocab[word]
    vec = W[idx]

    sims = W @ vec
    norms = np.linalg.norm(W, axis=1) * np.linalg.norm(vec)
    sims = sims / norms

    top_idx = np.argsort(sims)[::-1][1:top_n+1]

    return [(id_to_word[i], sims[i]) for i in top_idx]


def analogy(a, b, c, W, vocab, id_to_word, top_n=5):
    """
    Solve analogy: a - b + c = ?
    """

    for word in [a, b, c]:
        if word not in vocab:
            print(f"{word} not in vocabulary")
            return []

    vec = W[vocab[a]] - W[vocab[b]] + W[vocab[c]]

    sims = W @ vec
    norms = np.linalg.norm(W, axis=1) * np.linalg.norm(vec)
    sims = sims / norms

    top_idx = np.argsort(sims)[::-1][:top_n]

    return [(id_to_word[i], sims[i]) for i in top_idx]
```

Example usage:

```python
print("Most similar to 'king':")
print(most_similar("king", model.W, vocab, id_to_word))

print("Analogy: king - man + woman")
print(analogy("king", "man", "woman", model.W, vocab, id_to_word))
```

---

# 10. Possible Improvements

This implementation focuses on **clarity and understanding**, rather than maximum efficiency.

Possible improvements include:

- minibatch training
- vectorized updates
- subsampling of frequent words
- hierarchical softmax
- CBOW variant
- GPU acceleration

These techniques are used in optimized implementations such as **Gensim Word2Vec**.

---

# 11. References

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013)

Efficient Estimation of Word Representations in Vector Space

https://arxiv.org/abs/1301.3781