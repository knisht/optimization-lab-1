from typing import Tuple
from math import sqrt

from common.optimizer import Optimizer


class GoldenRatioOptimizer(Optimizer):
    fi = 0.5 + sqrt(5) / 2  # 1.618

    def _step(self, a: float, b: float) -> Tuple[float, float]:
        x1 = b - (b - a) / GoldenRatioOptimizer.fi
        x2 = a + (b - a) / GoldenRatioOptimizer.fi

        y1 = self.f(x1)
        y2 = self.f(x2)
        if y1 > y2:
            a = x1
        else:
            b = x2
        return a, b
