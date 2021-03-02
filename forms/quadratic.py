from typing import Tuple, Callable

import numpy as np
from scipy.stats import ortho_group


def __generate_eigenvalues(dim: int, scale: float) -> list[float]:
    min_eigen = np.random.random()
    max_eigen = min_eigen * scale
    if dim == 2:
        return [min_eigen, max_eigen]
    else:
        return sorted([min_eigen, *np.random.uniform(low=min_eigen, high=max_eigen, size=(dim - 2,)), max_eigen])


def __quadratic_form(matrix: np.ndarray) -> Callable[..., float]:
    def computable(arglist: list[float]) -> float:
        res = 0.0
        n, m = matrix.shape
        for i in range(n):
            for j in range(m):
                res += matrix[i][j] * arglist[i] * arglist[j]
        return res

    return lambda *x: computable(x)


def __quadratic_form_jacobi(matrix: np.ndarray) -> Callable[..., np.ndarray]:
    def computable(arglist: list[float]) -> np.ndarray:
        n, _ = matrix.shape
        result = np.zeros((n,))
        for i in range(n):
            result[i] = matrix[i].dot(arglist) * 2
        return result

    return lambda *x: computable(x)


def __make_symmetric(form: np.ndarray) -> np.ndarray:
    return (form + form.transpose()) / 2


def generate_random_form(condition_number: float, dim: int) -> Tuple[Callable[..., float], Callable[..., np.ndarray]]:
    random_matrix = ortho_group.rvs(dim=dim)
    eigen = __generate_eigenvalues(dim, condition_number)
    form = random_matrix @ np.diag(eigen) @ random_matrix.transpose()
    return __quadratic_form(form), __quadratic_form_jacobi(form)
