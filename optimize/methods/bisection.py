from typing import Tuple, Callable

from optimize.optimizer import Optimizer


class BisectionOptimizer(Optimizer):

    def _step(self, a: float, b: float) -> Tuple[float, float]:
        mid = (a + b) / 2.0
        return mid - self.eps / 3, mid + self.eps / 3

    def optimize(self) -> float:
        a = self.bounds[0]
        b = self.bounds[1]
        self.history = []
        while b - a > self.eps:
            self.n += 1
            a, b = self._step(a, b)
            self.history.append((a, b))
        return (a - b) / 2
