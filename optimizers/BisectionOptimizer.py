from typing import Tuple

from utils.optimizer import Optimizer


class BisectionOptimizer(Optimizer):
    def _step(self, a: float, b: float) -> Tuple[float, float]:
        """
        :return: new bounds
            example. the below implementation is a ternary search
        """
        if self.f(a) * self.f(b) > 0:
            raise Exception("No root found")
        else:
            midpoint = (a + b) / 2.0
            if self.f(a) * self.f(midpoint) < 0:  # Increasing but below 0 case
                b = midpoint
            else:
                a = midpoint
            return a, b
