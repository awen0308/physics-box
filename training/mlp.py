"""
Minimal Multi-Layer Perceptron implemented in pure NumPy.

No deep-learning framework is used on purpose: implementing forward pass,
ReLU, MSE loss, backpropagation and SGD by hand makes the math fully
transparent (which is exactly what a World Model interview wants to see).
"""

import numpy as np


class MLP:
    def __init__(self, layer_sizes, lr=0.01, seed=42, optimizer="adam"):
        self.rng = np.random.default_rng(seed)
        self.layer_sizes = list(layer_sizes)
        self.lr = lr
        self.optimizer = optimizer
        self.weights = []
        self.biases = []
        self.activations = []
        self.z_values = []
        self.loss_history = []

        # Adam 状态缓存（仅当使用 adam 优化器时生效）
        self.t = 0
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

        # Xavier / Glorot initialization
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            std = np.sqrt(2.0 / (in_dim + out_dim))
            self.weights.append(self.rng.normal(0.0, std, size=(out_dim, in_dim)))
            self.biases.append(np.zeros(out_dim))
            if optimizer == "adam":
                self.m_w.append(np.zeros((out_dim, in_dim)))
                self.v_w.append(np.zeros((out_dim, in_dim)))
                self.m_b.append(np.zeros(out_dim))
                self.v_b.append(np.zeros(out_dim))

    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    def forward(self, x):
        """Forward pass. x shape: (batch, input_dim). Returns (batch, output_dim)."""
        self.activations = [x]
        self.z_values = []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = self.activations[-1] @ W.T + b
            self.z_values.append(z)
            a = self.relu(z) if i < len(self.weights) - 1 else z
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, y_true):
        """Backward pass using MSE loss. Update rule depends on optimizer."""
        m = y_true.shape[0]
        grad = 2.0 * (self.activations[-1] - y_true) / m

        if self.optimizer == "adam":
            self.t += 1

        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = self.activations[i]
            dW = grad.T @ a_prev / m
            db = np.sum(grad, axis=0) / m

            if self.optimizer == "adam":
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dW
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dW ** 2)
                m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                self.weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)

                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db ** 2)
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
                self.biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
            else:
                self.weights[i] -= self.lr * dW
                self.biases[i] -= self.lr * db

            if i > 0:
                grad = grad @ self.weights[i]
                grad = grad * (self.z_values[i - 1] > 0).astype(float)

    def train(self, X, y, epochs, batch_size, X_val=None, y_val=None, patience=200):
        """SGD/Adam training with optional early stopping on a validation set.

        If X_val/y_val are given, training stops once the validation loss has
        not improved for `patience` consecutive epochs (best weights restored)."""
        n = X.shape[0]
        best_val = float("inf")
        best_weights = None
        best_biases = None
        stale = 0

        for epoch in range(epochs):
            indices = self.rng.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            total_loss = 0.0
            num_batches = 0
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                pred = self.forward(X_batch)
                loss = np.mean((pred - y_batch) ** 2)
                total_loss += loss
                num_batches += 1

                self.backward(y_batch)

            avg_loss = total_loss / max(num_batches, 1)
            self.loss_history.append(float(avg_loss))

            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = float(np.mean((val_pred - y_val) ** 2))
                if val_loss < best_val:
                    best_val = val_loss
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        self.weights = best_weights
                        self.biases = best_biases
                        print(f"Epoch {epoch:04d} | 早停 (验证集 {patience} 轮无提升), 最佳验证损失 {best_val:.8f}")
                        break

            if epoch % 100 == 0 or epoch == epochs - 1:
                extra = f" | Val: {best_val:.8f}" if X_val is not None else ""
                print(f"Epoch {epoch:04d} | Loss: {avg_loss:.8f}{extra}")

    @classmethod
    def from_json(cls, data):
        """Reconstruct a trained MLP from the JSON exported for Godot."""
        obj = cls([1])
        obj.weights = [np.array(layer["W"], dtype=float) for layer in data["layers"]]
        obj.biases = [np.array(layer["b"], dtype=float) for layer in data["layers"]]
        obj.layer_sizes = [w.shape[1] for w in obj.weights] + [obj.weights[-1].shape[0]]
        return obj
