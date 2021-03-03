from typing import Tuple, Callable

from optimize.optimizer import Optimizer


class FibonacciOptimizer(Optimizer):
    def __init__(self, f: Callable[[float], float], bounds: Tuple[float, float], eps: float, a, b, window):
        super().__init__(f, bounds, eps)
        self.F = []
        self.n = self.genF(a, b, window)

        self.l = [0.0] * self.n
        self.l[0] = a + self.F[self.n - 2] / self.F[self.n] * (b - a)

        self.mu = [0.0] * self.n
        self.mu[0] = a + self.F[self.n - 1] / self.F[self.n] * (b - a)

        self.a = [0.0] * self.n
        self.a[0] = a

        self.b = [0.0] * self.n
        self.b[0] = b

    def optimize(self) -> float:
        self.history = []
        n = self.n
        for k in range(self.n):
            self._log(self.a[k], self.b[k])
            if self.f(self.l[k]) > self.f(self.mu[k]):
                self.a[k + 1] = self.l[k]
                self.b[k + 1] = self.b[k]
                self.l[k + 1] = self.mu[k]
                self.mu[k + 1] = self.a[k + 1] + self.F[n - k - 1] / self.F[n - k] * (self.b[k + 1] - self.a[k + 1])
            else:
                self.a[k + 1] = self.a[k]
                self.b[k + 1] = self.mu[k]
                self.mu[k + 1] = self.l[k]
                self.l[k + 1] = self.a[k + 1] + self.F[n - k - 2] / self.F[n - k] * (self.b[k + 1] - self.a[k + 1])

            if k != self.n - 2:
                continue

            self.l[self.n] = self.l[self.n - 1]
            self.mu[self.n] = self.l[self.n] + self.eps

            if self.f(self.l[self.n]) == self.f(self.mu[n]):
                self.a[n] = self.mu[n]
                self.b[n] = self.b[n - 1]
            else:
                self.a[n] = self.a[n - 1]
                self.b[n] = self.mu[n]

        return 0.0

    def genF(self, a, b, l):
        min = (b - a) / l
        self.F = [0, 1]
        while self.F[-1] < min:
            self.F.append(self.F[-1] + self.F[-2])
        return len(self.F) - 1
