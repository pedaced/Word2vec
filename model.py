# model.py
import numpy as np

class SGNSModel:
    def __init__(self, vocab_size, embed_dim=50, lr=0.05):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = lr

        # embeddings
        self.W = np.random.randn(vocab_size, embed_dim) * 0.01
        self.U = np.random.randn(vocab_size, embed_dim) * 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def train_step_batch(self, centers, contexts, neg_samples):
        """
        centers: (B,)
        contexts: (B,)
        neg_samples: (B, k)
        """

        B = centers.shape[0]
        k = neg_samples.shape[1]

        # -----------------------
        # Embedding lookup
        # -----------------------
        V_c = self.W[centers]         # (B, D)
        U_o = self.U[contexts]        # (B, D)
        U_neg = self.U[neg_samples]   # (B, k, D)

        # -----------------------
        # Forward pass
        # -----------------------

        # positive scores
        score_pos = np.sum(V_c * U_o, axis=1)          # (B,)
        sig_pos = self.sigmoid(score_pos)

        # negative scores
        score_neg = np.sum(U_neg * V_c[:, None, :], axis=2)  # (B, k)
        sig_neg = self.sigmoid(-score_neg)

        # -----------------------
        # Loss
        # -----------------------
        loss = -np.sum(np.log(sig_pos + 1e-10)) - np.sum(np.log(sig_neg + 1e-10))

        # -----------------------
        # Gradients
        # -----------------------

        # positive gradients
        grad_pos = (sig_pos - 1)[:, None]  # (B,1)

        grad_v_pos = grad_pos * U_o        # (B,D)
        grad_uo = grad_pos * V_c           # (B,D)

        # negative gradients
        grad_neg = (1 - sig_neg)[:, :, None]   # (B,k,1)

        grad_v_neg = np.sum(grad_neg * U_neg, axis=1)   # (B,D)
        grad_un = grad_neg * V_c[:, None, :]            # (B,k,D)

        # total center gradient
        grad_v = grad_v_pos + grad_v_neg

        # -----------------------
        # Parameter updates
        # -----------------------

        # center embeddings
        np.add.at(self.W, centers, -self.lr * grad_v)

        # positive context
        np.add.at(self.U, contexts, -self.lr * grad_uo)

        # negative contexts
        np.add.at(self.U, neg_samples, -self.lr * grad_un)

        return loss / B