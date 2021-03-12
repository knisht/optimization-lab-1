from typing import Callable, Any, Tuple

import numpy as np

from common.oracle import Oracle
from optimize.multidimensional.MultiOptimizer import MultiOptimizer
from optimize.optimizer import Optimizer


def make_positive_definite(matrix: np.ndarray) -> np.ndarray:
    eigenvalues = np.linalg.eigvals(matrix)
    if all(eigenvalues > 0):
        return matrix
    else:
        min_eig = min(eigenvalues)
        additional_matrix = (-min_eig + 1.00000001) * np.eye(matrix.shape[0])
        return matrix + additional_matrix


class Newton(MultiOptimizer):

    def name(self) -> str:
        return "Newton"

    def iteration(self, f: Oracle, x: np.ndarray, optimizer: Callable[[Callable], Optimizer], iteration: int,
                  payload: Any) -> Tuple[np.ndarray, Any]:
        grad = f.grad(*x)
        # print(f"x = {x}, grad = {grad}")
        correct_hesse = make_positive_definite(f.hesse(*x))
        inv_hes = np.linalg.inv(correct_hesse)
        p = inv_hes.dot(grad)
        p = p / np.linalg.norm(p)

        # print(f"p = {p}")
        def g(lmbd):
            return f(*(x - p * lmbd))

        coeff = optimizer(g).optimize()
        # print(f"coeff = {coeff}")
        # print(f"prev_f = {f(*x)}, cur_f = {f(*(x - p * coeff))}")
        delta = p * coeff
        x1 = x - delta
        return x1, None