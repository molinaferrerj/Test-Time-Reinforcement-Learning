import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.W_out = np.random.randn(embedding_dim, vocab_size) * 0.1

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def train(self, training_pairs, epochs=5):
        for epoch in range(epochs):
            total_loss = 0
            
            for target_id, context_id in training_pairs:
                h = self.W_in[target_id]
                u = np.dot(h, self.W_out)
                y_pred = self._softmax(u)
                error = y_pred.copy()
                error[context_id] -= 1

                dW_out = np.outer(h, error)
                dW_in = np.dot(self.W_out, error)

                self.W_out -= self.lr * dW_out
                self.W_in[target_id] -= self.lr * dW_in

                total_loss -= np.log(y_pred[context_id] + 1e-9)
            
            avg_loss = total_loss / len(training_pairs)
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    def get_embedding(self, word_id):
        return self.W_in[word_id]