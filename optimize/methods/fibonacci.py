from typing import Tuple, Callable

from optimize.optimizer import Optimizer


class FibonacciOptimizer(Optimizer):
    def __init__(self, f: Callable[[float], float], bounds: Tuple[float, float], eps: float, a, b, window):
        super().__init__(f, bounds, eps)
        self.F = []
        self.n = self.genF(a, b, window)
        self.k = 0

        self.l = [0.0] * (self.n + 1)
        self.l[0] = a + self.F[self.n - 2] / self.F[self.n] * (b - a)

        self.mu = [0.0] * (self.n + 1)
        self.mu[0] = a + self.F[self.n - 1] / self.F[self.n] * (b - a)

        self.a = [0.0] * (self.n + 1)
        self.a[0] = a

        self.b = [0.0] * (self.n + 1)
        self.b[0] = b

    # TODO я поменял сигнатуру _step и optimize, предлагаю тут вообще _step выкинуть и просто переопределить optimize
    def _step(self, a: float, b: float) -> Tuple[float, float]:
        n = self.n
        k = self.k
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

        if self.k != self.n - 2:
            self.k = k + 1
            # TODO я добавил метод _log(a, b), надо вызывать его каждый раз, когда выбираешь новые границы
            # в начале optimize надо будет делать self.history = []
            return self.a[k], self.b[k]

        self.l[self.n] = self.l[self.n - 1]
        self.mu[self.n] = self.l[self.n] + self.eps

        if self.f(self.l[self.n]) == self.f(self.mu[n]):
            self.a[n] = self.mu[n]
            self.b[n] = self.b[n - 1]
        else:
            self.a[n] = self.a[n - 1]
            self.b[n] = self.mu[n]

        return self.a[k + 1], self.b[k + 1]

    def genF(self, a, b, l):
        min = (b - a) / l
        self.F = [0, 1]
        while self.F[-1] < min:
            self.F = self.F[-1] + self.F[-2]
        return len(self.F) - 1
