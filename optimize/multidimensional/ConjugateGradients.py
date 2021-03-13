from typing import Callable, Any, Tuple

import numpy as np

from common.oracle import Oracle
from optimize.multidimensional.MultiOptimizer import MultiOptimizer
from optimize.optimizer import Optimizer


class ConjugateGradients(MultiOptimizer):
    def init(self) -> Any:
        return [], []

    def name(self) -> str:
        return "Conjugate gradients"

    def iteration(self, f: Oracle, x: np.ndarray,
                  optimizer: Callable[[Callable], Optimizer], iteration: int, payload: Any) -> Tuple[np.ndarray, Any]:
        ws, ps = payload

        def __square(w: np.ndarray) -> float:
            return w.dot(w)

        ws.append(-f.grad(*x))  # w_k
        gamma = 0.0 if iteration % len(x) == 0 else __square(ws[-1]) / __square(ws[-2])
        p = ws[-1]
        p = p #/ np.linalg.norm(p)
        if gamma != 0.0:
            p += gamma * ps[-1]
        ps.append(p)

        def g(lmbd):
            return f(*(x + p * lmbd))

        lr = optimizer(g).optimize()
        delta = p * lr
        # print(f"latest gradient(w): {ws[-1]}, p = {p}, delta = {delta}, x = {x}, gamma = {gamma}, lr = {lr}")
        x1 = x + delta
        return x1, [ws, ps]
