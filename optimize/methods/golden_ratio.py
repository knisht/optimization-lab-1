from typing import Tuple, Callable
from math import sqrt

from optimize.optimizer import Optimizer


class GoldenRatioOptimizer(Optimizer):
    fi = 0.5 + sqrt(5) / 2  # 1.618

    def _step(self, a: float, b: float) -> Tuple[float, float]:
        x1 = b - (b - a) / GoldenRatioOptimizer.fi
        x2 = a + (b - a) / GoldenRatioOptimizer.fi
        return x1, x2

    def optimize_lin(self) -> float:
        a = self.bounds[0]
        b = self.bounds[1]
        self.history = []
        self.n = 0
        while b - a > self.eps:
            self.n += 1
            a, b = self._step(a, b)
            self.history.append((a, b))
        return (a + b) / 2
