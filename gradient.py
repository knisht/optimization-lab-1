from typing import Callable, List, Tuple, Union, Optional

import numpy as np

from OptimizationResult import OptimizationResult
from common.oracle import Oracle
from optimize.optimizer import Optimizer


def gradient_descent(
        f: Oracle, x0: np.ndarray, step_optimizer: Callable[[Callable], Optimizer],
        iterations: Optional[int] = None, dx: Optional[float] = None, df: Optional[float] = None
) -> OptimizationResult:
    x = x0
    it = 0
    trajectory = [x]
    cause = None
    while (iterations is None) or (it < iterations):
        # print("grad x: ", x)
        grad = f.grad(*x)

        def g(lmbd):
            return f(*(x - grad * lmbd))

        delta = grad * step_optimizer(g).optimize()
        x1 = x - delta
        trajectory.append(x1)
        if dx is not None and np.linalg.norm(delta) < dx:
            cause = "Exceed limit of accuracy for argument"
            break
        if df is not None and abs(f(*x1) - f(*x)) < df:
            cause = "Exceed limit of accuracy for function"
            break
        if x.dot(x) > 1e10:
            cause = "Divergence"
            break
        it += 1
        x = x1

    if cause is None:
        cause = "Exceed limit of iterations"
    return OptimizationResult(x, it, trajectory, cause)


def linear_search(f: Oracle, bounds: Tuple[np.ndarray, np.ndarray], dx: float) -> np.ndarray:
    x, x1 = bounds[0], np.zeros((f.n,))
    v = f(*x)

    def iterate(dim: int):
        nonlocal x, x1, v
        if dim == f.n:
            v1 = f(*x1)
            if v1 < v:
                x, v = x1.copy(), v1
        else:
            ticks = np.arange(bounds[0][dim], bounds[1][dim], dx)
            for i in ticks:
                x1[dim] = i
                iterate(dim + 1)

    iterate(0)
    return x
