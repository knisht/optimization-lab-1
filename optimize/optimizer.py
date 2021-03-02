from typing import Callable, Tuple


class Optimizer:
    def __init__(self, f: Callable[[float], float], bounds: Tuple[float, float], eps: float):
        self.f = f
        self.bounds = bounds
        self.eps = eps

    def _step(self, a: float, b: float) -> Tuple[float, float]:
        """
        :return: new potential bounds
            example. the default implementation is a ternary search
        """
        return (2 * a + b) / 3, (a + 2 * b) / 3

    def optimize(self) -> float:
        a, b = self.bounds[0], self.bounds[1]
        while b - a > self.eps:
            x1, x2 = self._step(a, b)
            if self.f(x1) < self.f(x2):
                b = x2
            else:
                a = x1
        return (a + b) / 2
