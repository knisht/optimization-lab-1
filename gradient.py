from typing import Callable, List, Tuple, Union

import numpy as np

from utils.oracle import Oracle
from utils.optimizer import Optimizer


def gradient_descent(
        f: Oracle, x0: np.ndarray, step_optimizer: Callable[..., Optimizer],
        iterations: int = 1000, eps: float = 1e-6,
) -> np.ndarray:
    x = x0
    it = 0

    while it < iterations:
        grad = f.grad(*x)

        def g(lmbd):
            return f(*(x - grad * lmbd))

        delta = grad * step_optimizer(g, (0, 2), eps).optimize()
        x -= delta
        if np.linalg.norm(delta) < eps:
            break
        it += 1

    return x
