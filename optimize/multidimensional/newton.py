from typing import Callable, Any, Tuple

import numpy as np

from common.oracle import Oracle
from common.stats import StatCollector
from optimize.multidimensional.multioptimizer import MultiOptimizer
from optimize.unidimensional.optimizer import Optimizer


def make_positive_definite(matrix: np.ndarray, stats: StatCollector) -> np.ndarray:
    eigenvalues = np.linalg.eigvals(matrix)
    stats.report(f'eigenvalues {matrix.shape[0]}', 1)
    # todo ну вообще непонятно, как это руками посчитать, так что можно отдельным пунктом выводить

    if all(eigenvalues > 0):
        return matrix
    else:
        min_eig = min(eigenvalues)
        additional_matrix = (-min_eig + 1.00000001) * np.eye(matrix.shape[0])
        stats.report('+', 1 + matrix.shape[0] ** 2)
        stats.report('*', matrix.shape[0] ** 2)
        return matrix + additional_matrix


class Newton(MultiOptimizer):
    def name(self) -> str:
        return "Newton"

    def iteration(
            self, f: Oracle, x: np.ndarray, optimizer: Callable[[Callable], Optimizer],
            iteration: int, payload: Any
    ) -> Tuple[np.ndarray, Any]:
        grad = f.grad(*x)
        # print(f"x = {x}, grad = {grad}")
        correct_hesse = make_positive_definite(f.hesse(*x), self._stats)
        inv_hes = np.linalg.inv(correct_hesse)
        p = inv_hes.dot(grad)

        # p = p / np.linalg.norm(p)

        # print(f"p = {p}")
        def g(lmbd):
            return f(*(x - p * lmbd))

        opt = optimizer(g)
        opt._stats = self._stats
        coeff = opt.optimize()
        # print(f"coeff = {coeff}")
        # print(f"prev_f = {f(*x)}, cur_f = {f(*(x - p * coeff))}")
        delta = p * coeff
        self._stats.report('*', p.size)
        x1 = x - delta
        self._stats.report('-', x.size)

        return x1, None
