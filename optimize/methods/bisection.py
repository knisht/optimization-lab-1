from typing import Tuple

from common.optimizer import Optimizer


class BisectionOptimizer(Optimizer):
    def _step(self, a: float, b: float) -> Tuple[float, float]:
        if self.f(a) * self.f(b) > 0:
            raise Exception("No root found")
        else:
            mid = (a + b) / 2.0
            if self.f(a) * self.f(mid) < 0:
                b = mid
            else:
                a = mid
        return a, b
