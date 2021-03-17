from typing import Callable, Any, Tuple

import numpy as np

from common.oracle import Oracle
from common.stats import StatCollector
from optimize.multidimensional.multioptimizer import MultiOptimizer
from optimize.unidimensional.optimizer import Optimizer


class GradientDescent(MultiOptimizer):
    def name(self) -> str:
        return "Gradient descent"

    def iteration(
            self, f: Oracle, x: np.ndarray, optimizer: Callable[[Callable], Optimizer],
            iteration: int, payload: Any
    ) -> Tuple[np.ndarray, Any]:
        pregrad = f.grad(*x)
        grad = pregrad  # / np.linalg.norm(pregrad)

        # print(f"point: {x}, grad: {grad} | {pregrad}")

        def g(lmbd):
            return f(*(x - grad * lmbd))

        opt = optimizer(g)
        opt._stats = self._stats
        coeff = optimizer(g).optimize()

        delta = grad * coeff
        self._stats.report('*', grad.size)
        # print(f"delta: {delta}, coeff: {coeff}")
        x1 = x - delta
        self._stats.report('-', x.size)

        return x1, None
