from typing import Callable, Optional, Any, Tuple

import numpy as np

from OptimizationResult import OptimizationResult
from common.oracle import Oracle
from optimize.multidimensional.MultiOptimizer import MultiOptimizer
from optimize.optimizer import Optimizer


class GradientDescent(MultiOptimizer):
    def iteration(self, f: Oracle, x: np.ndarray,
                  optimizer: Callable[[Callable], Optimizer],
                  iteration: int, payload: Any) -> Tuple[np.ndarray, Any]:
        grad = f.grad(*x)

        def g(lmbd):
            return f(*(x - grad * lmbd))

        delta = grad * optimizer(g).optimize()
        x1 = x - delta

        return x1, None
