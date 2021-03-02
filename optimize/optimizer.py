from typing import Callable, Tuple


class Optimizer:
    def __init__(self, f: Callable[[float], float], bounds: Tuple[float, float], eps: float):
        self.f = f
        self.bounds = bounds
        self.eps = eps

    def _step(self, a: float, b: float) -> Tuple[float, float]:
        """
        :return: new bounds
            example. the below implementation is a ternary search
        """
        x, y = (2 * a + b) / 3, (a + 2 * b) / 3
        if self.f(x) < self.f(y):
            return a, y
        else:
            return b, x

    def optimize(self) -> float:
        a, b = self.bounds[0], self.bounds[1]
        while b - a > self.eps:
            a, b = self._step(a, b)
        return (a + b) / 2
