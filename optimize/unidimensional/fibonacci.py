from typing import Tuple, Callable

from optimize.unidimensional.optimizer import Optimizer


class FibonacciOptimizer(Optimizer):
    def __init__(self, f: Callable[[float], float], bounds: Tuple[float, float], eps: float):
        super().__init__(f, bounds, eps)
        self.F = []

        a0 = bounds[0]
        b0 = bounds[1]

        self.n = self.genF(a0, b0, eps)

    def optimize(self) -> float:
        a = self.bounds[0]
        b = self.bounds[1]
        self.history = [(a, b)]
        x1 = a + self.F[self.n] / self.F[self.n + 2] * (b - a)
        x2 = a + self.F[self.n + 1] / self.F[self.n + 2] * (b - a)
        f1, f2 = self.f(x1), self.f(x2)
        self.f_calls = 1
        for i in range(1, self.n):
            if f2 < f1:
                f1 = f2
                a = x1
                x1 = x2
                x2 = a + self.F[self.n + 1 - i] / self.F[self.n + 2 - i] * (b - a)
                f2 = self.f(x2)
            else:
                f2 = f1
                b = x2
                x2 = x1
                x1 = a + self.F[self.n - i] / self.F[self.n + 2 - i] * (b - a)
                f1 = self.f(x1)
            self.f_calls += 1
            self._log(a, b)
        return (a + b) / 2

    def genF(self, a, b, l):
        min = (b - a) / l
        self.F = [1, 1,]
        n = 2
        while self.F[-1] < min:
            self.F.append(self.F[-1] + self.F[-2])
        n = len(self.F) - 1
        self.F.append(self.F[-1] + self.F[-2])
        self.F.append(self.F[-1] + self.F[-2])
        self.F.append(self.F[-1] + self.F[-2])
        return n