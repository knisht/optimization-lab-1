from typing import Callable, Tuple, List


class Optimizer:
    def __init__(self, f: Callable[[float], float], bounds: Tuple[float, float], eps: float):
        self.f = f
        self.bounds = bounds
        self.eps = eps
        self.n = 0
        self.history: List[Tuple[float, float]] = []
        self.f_calls = 0

    def _log(self, a: float, b: float):
        self.history.append((a, b))

    def _step(self, a: float, b: float) -> Tuple[float, float]:
        """
        :return: new potential bounds
            example. the default implementation is a ternary search
        """
        return (2 * a + b) / 3, (a + 2 * b) / 3

    def optimize(self) -> float:
        self.history = []
        a, b = self.bounds[0], self.bounds[1]
        self._log(a, b)

        while b - a > self.eps:
            self.f_calls += 2
            x1, x2 = self._step(a, b)
            if self.f(x1) < self.f(x2):
                b = x2
            else:
                a = x1
            # print(a, b)
            self._log(a, b)

        return (a + b) / 2

    def stats(self):
        return {
            'steps': len(self.history),
            'history': self.history
        }
