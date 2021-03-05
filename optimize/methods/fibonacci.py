from typing import Tuple, Callable

from optimize.optimizer import Optimizer


class FibonacciOptimizer(Optimizer):
    def __init__(self, f: Callable[[float], float], bounds: Tuple[float, float], eps: float):
        super().__init__(f, bounds, eps)
        self.F = []

        a0 = bounds[0]
        b0 = bounds[1]

        self.n = self.genF(a0, b0, eps)

        self.l = [0.0] * self.n
        self.l[0] = a0 + self.F[self.n - 2] / self.F[self.n - 1] * (b0 - a0)

        self.mu = [0.0] * self.n
        self.mu[0] = a0 + self.F[self.n - 1] / self.F[self.n - 1] * (b0 - a0)

        self.a = [0.0] * self.n
        self.a[0] = a0

        self.b = [0.0] * self.n
        self.b[0] = b0

    def optimize(self) -> float:
        self.history = []
        lst = self.n - 1
        for k in range(self.n - 1):
            self._log(self.a[k], self.b[k])
            if self.f(self.l[k]) > self.f(self.mu[k]):
                self.a[k + 1] = self.l[k]
                self.b[k + 1] = self.b[k]
                self.l[k + 1] = self.mu[k]
                self.mu[k + 1] = self.a[k + 1] + self.F[lst - k - 1] / self.F[lst - k] * (
                            self.b[k + 1] - self.a[k + 1])
            else:
                self.a[k + 1] = self.a[k]
                self.b[k + 1] = self.mu[k]
                self.mu[k + 1] = self.l[k]
                self.l[k + 1] = self.a[k + 1] + self.F[lst - k - 2] / self.F[lst - k] * (
                            self.b[k + 1] - self.a[k + 1])

            if k != lst - 2:
                continue

            self.l[lst] = self.l[lst]
            self.mu[lst] = self.l[lst] + self.eps

            if self.f(self.l[lst]) == self.f(self.mu[lst]):
                self.a[lst] = self.mu[lst]
                self.b[lst] = self.b[lst]
            else:
                self.a[lst] = self.a[lst]
                self.b[lst] = self.mu[lst]

        self._log(self.a[lst], self.b[lst])
        return (self.a[lst] + self.b[lst]) / 2

    def genF(self, a, b, l):
        min = (b - a) / l
        self.F = [0, 1]
        while self.F[-1] < min:
            self.F.append(self.F[-1] + self.F[-2])
        return len(self.F)

    def optimize_lin(self) ->float:
        return self.optimize()