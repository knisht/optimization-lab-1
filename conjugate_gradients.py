from typing import Callable, List, Tuple, Union, Optional

import numpy as np

from OptimizationResult import OptimizationResult
from common.oracle import Oracle
from optimize.optimizer import Optimizer


def __square(w: np.ndarray) -> float:
    return w.dot(w)


def conjugate_gradients(
        f: Oracle, x0: np.ndarray, step_optimizer: Callable[[Callable], Optimizer],
        iterations: Optional[int] = None, dx: Optional[float] = None, df: Optional[float] = None
) -> OptimizationResult:
    x = x0
    it = 0
    trajectory = [x]
    ws = []
    ps = []
    cause = None
    while (iterations is None) or (it < iterations):
        ws.append(-f.grad(*x))  # w_k
        gamma = 0.0 if it % len(x0) == 0 else __square(ws[-1]) / __square(ws[-2])
        p = ws[-1]
        if gamma != 0.0:
            p += gamma * ps[-1]
        ps.append(p)

        def g(lmbd):
            return f(*(x + p * lmbd))

        # print(ws[-1])
        delta = p * step_optimizer(g).optimize()
        x1 = x + delta
        # print(f"current point: {x};\nw: {ws[-1]};\ndelta:{delta};\n")

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
