from typing import Callable, List, Tuple, Union, Optional

import numpy as np

from common.oracle import Oracle
from optimize.optimizer import Optimizer


def gradient_descent(
        f: Oracle, x0: np.ndarray, step_optimizer: Callable[[Callable], Optimizer],
        iterations: Optional[int] = None, dx: Optional[float] = None, df: Optional[float] = None
) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    x = x0
    it = 0
    trajectory = [x]
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


def linear_search(f: Oracle, bounds: Tuple[np.ndarray, np.ndarray], dx: float) -> float:
    x, x1 = bounds[0], np.zeros((f.dim,))
    v = f(*x)

    def iterate(dim: int):
        nonlocal x, x1, v
        if dim == oracle.n:
            v1 = f(*x1)
            if v1 < v:
                x, v = x1, v1
        else:
            ticks = np.arange(bounds[0][dim], bounds[1][dim], dx)
            for i in ticks:
                x1[dim] = i
                iterate(dim + 1)

    iterate(0)
    return x
