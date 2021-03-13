from typing import Callable, Any, Tuple

import numpy as np

from common.oracle import Oracle
from optimize.multidimensional.MultiOptimizer import MultiOptimizer
from optimize.optimizer import Optimizer


class GradientDescent(MultiOptimizer):

    def name(self) -> str:
        return "Gradient descent"

    def iteration(self, f: Oracle, x: np.ndarray,
                  optimizer: Callable[[Callable], Optimizer],
                  iteration: int, payload: Any) -> Tuple[np.ndarray, Any]:
        pregrad = f.grad(*x)
        grad = pregrad #/ np.linalg.norm(pregrad)
        # print(f"point: {x}, grad: {grad} | {pregrad}")

        def g(lmbd):
            return f(*(x - grad * lmbd))

        coeff = optimizer(g).optimize()
        delta = grad * coeff
        # print(f"delta: {delta}, coeff: {coeff}")
        x1 = x - delta

        return x1, None
