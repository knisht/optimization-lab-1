from typing import Callable, List, Tuple, Union, Optional

import numpy as np

from OptimizationResult import OptimizationResult
from common.oracle import Oracle
from optimize.optimizer import Optimizer


def make_positive_definite(matrix: np.ndarray) -> np.ndarray:
    eigenvalues = np.linalg.eigvals(matrix)
    if all(eigenvalues > 0):
        return matrix
    else:
        min_eig = min(eigenvalues)
        additional_matrix = (-min_eig + 1.00000001) * np.eye(matrix.shape[0])
        return matrix + additional_matrix

def newton(
        f: Oracle, x0: np.ndarray, step_optimizer: Callable[[Callable], Optimizer],
        iterations: Optional[int] = None, dx: Optional[float] = None, df: Optional[float] = None
) -> OptimizationResult:
    x = x0
    it = 0
    trajectory = [x]
    cause = None
    while (iterations is None) or (it < iterations):
        grad = f.grad(*x)
        correct_hesse = make_positive_definite(f.hesse(*x))
        inv_hes = np.linalg.inv(correct_hesse)
        p = inv_hes.dot(grad)
        def g(lmbd):
            return f(*(x - p * lmbd))

        coeff = step_optimizer(g).optimize()
        delta = p * coeff
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
