from typing import Callable, List, Tuple, Union

import numpy as np

from common.oracle import Oracle
from optimize.optimizer import Optimizer


def gradient_descent(
        f: Oracle, x0: np.ndarray, step_optimizer: Callable[[Callable], Optimizer],
        iterations: int = None, dx: Union[None, float] = None, df: Union[None, float] = None
) -> Tuple[np.ndarray, int, list[np.ndarray]]:
    x = x0
    it = 0
    trajectory = []
    while (iterations is None) or (it < iterations):
        grad = f.grad(*x)

        def g(lmbd):
            return f(*(x - grad * lmbd))

        delta = grad * step_optimizer(g).optimize()
        x1 = x - delta
        trajectory.append(x1)
        if dx is not None and np.linalg.norm(delta) < dx:
            break
        if df is not None and abs(f(*x1) - f(*x)) < df:
            break
        it += 1
        x = x1

    return x, it, trajectory
