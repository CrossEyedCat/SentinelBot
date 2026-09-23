"""A small multilayer perceptron in numpy, with Adam - the student policy.

numpy rather than a deep learning framework on purpose. The network is 27 -> 64 -> 64 -> 3, about
6k parameters; training it takes seconds on a CPU, and more importantly the robot then needs
nothing at runtime that the rest of the stack does not already depend on. Adding a torch runtime to
a Raspberry Pi payload to evaluate 6k parameters would be the tail wagging the dog.

Outputs are tanh-bounded and scaled by the platform limits, so the policy cannot ask for a velocity
the base could not deliver, whatever the training data looked like.
"""
import numpy as np


class MLP:

    def __init__(self, sizes=(27, 64, 64, 3), seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.sizes = tuple(sizes)
        self.w, self.b = [], []
        for a, c in zip(self.sizes, self.sizes[1:]):
            self.w.append(rng.normal(0.0, np.sqrt(1.0 / a), (a, c)))
            self.b.append(np.zeros(c))
        self._m = [np.zeros_like(p) for p in self.w + self.b]
        self._v = [np.zeros_like(p) for p in self.w + self.b]
        self._t = 0

    # ------------------------------------------------------------------ inference
    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.atleast_2d(np.asarray(x, dtype=float))
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            h = h @ w + b
            h = np.tanh(h)          # the last layer is tanh too: the output is bounded by design
        return h if np.ndim(x) > 1 else h[0]

    # ------------------------------------------------------------------ training
    def _forward(self, x):
        acts = [x]
        h = x
        for w, b in zip(self.w, self.b):
            h = np.tanh(h @ w + b)
            acts.append(h)
        return acts

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray):
        n = len(x)
        acts = self._forward(x)
        out = acts[-1]
        diff = out - y
        loss = float((diff ** 2).mean())

        gw = [None] * len(self.w)
        gb = [None] * len(self.b)
        delta = (2.0 / (n * y.shape[1])) * diff * (1.0 - out ** 2)
        for i in range(len(self.w) - 1, -1, -1):
            gw[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i:
                delta = (delta @ self.w[i].T) * (1.0 - acts[i] ** 2)
        return loss, gw + gb

    def adam(self, grads, lr: float = 3e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self._t += 1
        params = self.w + self.b
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g * g
            mh = self._m[i] / (1 - b1 ** self._t)
            vh = self._v[i] / (1 - b2 ** self._t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 40, batch: int = 256,
            lr: float = 3e-3, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        loss = float('nan')
        for ep in range(epochs):
            order = rng.permutation(len(x))
            for i in range(0, len(x), batch):
                sel = order[i:i + batch]
                loss, grads = self.loss_and_grads(x[sel], y[sel])
                self.adam(grads, lr=lr)
        return loss

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        blob = {'sizes': np.asarray(self.sizes)}
        blob.update({f'w{i}': w for i, w in enumerate(self.w)})
        blob.update({f'b{i}': b for i, b in enumerate(self.b)})
        np.savez(path, **blob)

    @staticmethod
    def load(path: str) -> 'MLP':
        blob = np.load(path)
        net = MLP(tuple(int(v) for v in blob['sizes']))
        net.w = [blob[f'w{i}'] for i in range(len(net.w))]
        net.b = [blob[f'b{i}'] for i in range(len(net.b))]
        return net
