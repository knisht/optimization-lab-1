from typing import Tuple
from math import sqrt

from common.optimizer import Optimizer


class GoldenRatioOptimizer(Optimizer):
    fi = 0.5 + sqrt(5) / 2  # 1.618

    def _step(self, a: float, b: float) -> Tuple[float, float]:
        x1 = b - (b - a) / GoldenRatioOptimizer.fi
        x2 = a + (b - a) / GoldenRatioOptimizer.fi
        return x1, x2
