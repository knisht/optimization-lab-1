from typing import Tuple

from utils.optimizer import Optimizer


class GoldenSectionOptimizer(Optimizer):
    def _step(self, a: float, b: float) -> Tuple[float, float]:
        """
        :return: new bounds
            example. the below implementation is a ternary search
        """
        fi = 1.618
        x1 = b - (b - a) / fi
        x2 = a + (b - a) / fi

        y1 = self.f(x1)
        y2 = self.f(x2)
        if y1 > y2:
            a = x1
        else:
            b = x2
        return a, b
