# Word2Vec (Skip-Gram with Negative Sampling) — NumPy Implementation

This project implements **Word2Vec from scratch using only NumPy**.

The goal is to implement the **core training loop of Word2Vec**, including:

- forward pass
- loss computation
- gradient derivation
- parameter updates

No deep learning frameworks (PyTorch, TensorFlow, etc.) are used.

The model implemented is **Skip-Gram with Negative Sampling (SGNS)** as introduced by **Mikolov et al. (2013)**.

---

# 1. Project Structure

```
.
├── train.py          # training pipeline
├── model.py          # SGNS model implementation
├── utils.py          # preprocessing and evaluation utilities
├── requirements.txt
└── README.md
```

### File Responsibilities

**train.py**

Responsible for:

- loading dataset
- preprocessing tokens
- generating skip-gram pairs
- training the model
- running evaluation

**model.py**

Contains the implementation of the **SGNSModel**, including:

- embedding matrices
- forward pass
- loss computation
- gradient calculation
- parameter updates

**utils.py**

Contains helper functions:

- vocabulary construction
- skip-gram generation
- negative sampling distribution
- similarity queries

---

# 2. Word2Vec Overview

Word2Vec learns **dense vector representations of words** such that words appearing in similar contexts have similar embeddings.

Example relationships learned by Word2Vec:

```
king - man + woman ≈ queen
```

or

```
king ≈ queen
car ≈ vehicle
```

The idea is that **semantic and syntactic information emerges from context prediction**.

---

# 3. Skip-Gram with Negative Sampling

The skip-gram model predicts **context words given a center word**.

For a center word \( w_c \) and context word \( w_o \), the objective function is:

\[
\log \sigma(u_o^T v_c) + \sum_{k=1}^{K} \log \sigma(-u_k^T v_c)
\]

Where:

- \( v_c \) = center word embedding
- \( u_o \) = context embedding
- \( u_k \) = negative sample embeddings
- \( \sigma \) = sigmoid function
- \( K \) = number of negative samples

The model learns **two embedding matrices**:

```
W : center word embeddings   (vocab_size × embedding_dim)
U : context word embeddings  (vocab_size × embedding_dim)
```

---

# 4. Dataset

The model uses the **text8 dataset**, loaded via:

```
gensim.downloader
```

To keep training manageable on CPU, the dataset is truncated to:

```
1,000,000 tokens
```

---

# 5. Training Pipeline

The training pipeline implemented in `train.py` consists of the following steps.

---

## Step 1 — Dataset Loading

```
corpus = api.load("text8")
tokens = [word for sentence in corpus for word in sentence]
```

The corpus is flattened into a sequence of tokens.

---

## Step 2 — Vocabulary Construction

The vocabulary is created using token frequency counts.

```
vocab[word] → integer ID
id_to_word[id] → word
```

The token sequence is then converted to **integer indices**.

This allows fast embedding lookups.

---

## Step 3 — Skip-Gram Pair Generation

Skip-gram training pairs are generated using a sliding window.

Example with window size = 2:

```
the quick brown fox jumps
```

Center word: **brown**

Context words:

```
quick
the
fox
jumps
```

Each pair becomes a **training example**.

---

## Step 4 — Negative Sampling Distribution

Negative words are sampled according to:

\[
P(w) \propto f(w)^{0.75}
\]

Where:

- \( f(w) \) = word frequency

This distribution reduces the probability of extremely frequent words dominating training.

---

# 6. Model Implementation

The SGNS model is implemented in **model.py**.

### Embedding initialization

Two matrices are initialized:

```
self.W → center embeddings
self.U → context embeddings
```

Both are initialized with small random values.

---

# 7. Vectorized Batch Training

A key design choice in this implementation is **vectorized batch training**.

Instead of updating one training example at a time, the model processes **batches of skip-gram pairs**.

This significantly improves performance.

Batch dimensions:

```
centers      (B)
contexts     (B)
neg_samples  (B, K)
```

Embedding lookups produce:

```
V_c    (B, D)
U_o    (B, D)
U_neg  (B, K, D)
```

Where:

```
B = batch size
D = embedding dimension
K = number of negative samples
```

---

# 8. Forward Pass

Positive score:

```
score_pos = dot(v_c , u_o)
```

Vectorized form:

```
score_pos = sum(V_c * U_o)
```

Negative scores:

```
score_neg = dot(v_c , u_k)
```

Vectorized:

```
score_neg = sum(U_neg * V_c)
```

Sigmoid is applied to compute probabilities.

---

# 9. Loss Function

The batch loss is:

```
L = - log(sigmoid(score_pos))
    - sum log(sigmoid(-score_neg))
```

This encourages:

- high similarity for true context pairs
- low similarity for negative samples

---

# 10. Gradient Computation

Gradients are derived analytically from the SGNS objective.

For positive pairs:

```
grad = sigmoid(score_pos) - 1
```

For negative samples:

```
grad = 1 - sigmoid(-score_neg)
```

Gradients are then propagated to update:

- center embeddings
- positive context embeddings
- negative context embeddings

---

# 11. Efficient Parameter Updates

Parameter updates use **NumPy's indexed accumulation**:

```
np.add.at(...)
```

This allows safe updates when **the same word appears multiple times in a batch**.

---

# 12. Training Configuration

| Parameter | Value |
|--------|------|
| embedding dimension | 50 |
| window size | 2 |
| negative samples | 3 |
| batch size | 512 |
| epochs | 3 |
| learning rate | 0.05 |

---

# 13. Running Training

Install dependencies:

```
pip install -r requirements.txt
```

Run training:

```
python train.py
```

Example training output:

```
Epoch 1 | Batch 100 | Avg Loss: 2.10
Epoch 1 | Batch 200 | Avg Loss: 2.03
```

---

# 14. Evaluating Embeddings

After training, embeddings can be inspected using cosine similarity.

Example:

```
Most similar words to "king":

queen
prince
monarch
emperor
throne
```

Evaluation function:

```
most_similar(word, W, vocab, id_to_word)
```

---

# 15. Possible Extensions

Possible improvements to the current implementation:

- subsampling frequent words
- CBOW variant
- hierarchical softmax
- larger datasets
- GPU acceleration

---

# 16. References

Tomas Mikolov et al. (2013)

Efficient Estimation of Word Representations in Vector Space

https://arxiv.org/abs/1301.3781