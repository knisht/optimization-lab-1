from typing import Callable, Any, Tuple

import numpy as np

from common.oracle import Oracle
from optimize.multidimensional.multioptimizer import MultiOptimizer
from optimize.unidimensional.optimizer import Optimizer


class ConjugateGradients(MultiOptimizer):
    def init(self) -> Any:
        return [], []

    def name(self) -> str:
        return "Conjugate gradients"

    def iteration(
            self, f: Oracle, x: np.ndarray, optimizer: Callable[[Callable], Optimizer],
            iteration: int, payload: Any
    ) -> Tuple[np.ndarray, Any]:
        ws, ps = payload

        def __square(w: np.ndarray) -> float:
            self._stats.report('*', w.size)
            self._stats.report('+', w.size - 1)
            return w.dot(w)

        ws.append(-f.grad(*x))  # w_k
        gamma = 0.0
        if iteration % len(x) != 0:
            gamma = __square(ws[-1]) / __square(ws[-2])
            self._stats.report('/', 1)

        p = ws[-1]
        # p = p  # / np.linalg.norm(p)
        if gamma != 0.0:
            p += gamma * ps[-1]
            self._stats.report('*', p.size)
            self._stats.report('+', p.size)
        ps.append(p)

        def g(lmbd):
            return f(*(x + p * lmbd))

        opt = optimizer(g)
        opt._stats = self._stats
        lr = opt.optimize()

        delta = p * lr
        self._stats.report('*', p.size)
        # print(f"latest gradient(w): {ws[-1]}, p = {p}, delta = {delta}, x = {x}, gamma = {gamma}, lr = {lr}")
        x1 = x + delta
        self._stats.report('+', x.size)

        return x1, [ws, ps]
