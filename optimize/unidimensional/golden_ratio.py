from typing import Tuple, Callable
from math import sqrt

from optimize.unidimensional.optimizer import Optimizer


class GoldenRatioOptimizer(Optimizer):
    phi = 0.5 + sqrt(5) / 2  # 1.618
    resphi = 2 - phi

    def _step(self, a: float, b: float) -> Tuple[float, float]:
        x1 = b - (b - a) / GoldenRatioOptimizer.phi
        x2 = a + (b - a) / GoldenRatioOptimizer.phi
        return x1, x2

    def optimize(self) -> float:
        a = self.bounds[0]
        b = self.bounds[1]
        self.history = [(a, b)]
        self.n = 0

        x1, x2 = self._step(a, b)
        f1 = self.f(x1)
        f2 = self.f(x2)
        self.f_calls = 1
        while abs(b - a) > self.eps:
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + self.resphi * (b - a)
                f1 = self.f(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = b - self.resphi * (b - a)
                f2 = self.f(x2)
            self.f_calls += 1
            self.history.append((a, b))

        return (x1 + x2) / 2
