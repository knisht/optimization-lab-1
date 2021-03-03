from typing import List

import numpy as np

from common.oracle import Oracle
from forms.quadratic import generate_random_form
from gradient import gradient_descent
from optimize.optimizer import Optimizer
from matplotlib import pyplot as plt


def generate_graph(n: int, ks: List[float]):
    iterations = []
    for k in ks:
        form, jacobi = generate_random_form(k, n)
        oracle = Oracle(n, form, jacobi)

        def ternary_search(g):
            return Optimizer(g, (0, 2), 0.001)

        initial_point = np.random.uniform(low=0.001, high=1.0, size=(n,))
        argmin, iteration, _ = gradient_descent(f=oracle, x0=initial_point, step_optimizer=ternary_search, df=1e-7,
                                                dx=1e-7)
        iterations.append(iteration)

    plt.grid(linestyle='--')
    plt.plot(ks, iterations, linestyle='-', marker='.', label=f'T({n}, k)')
    plt.xlabel('Condition number')
    plt.ylabel(f'T({n}, k)')
    plt.savefig(f"results/quadratic_{n}.png")
    plt.show()


def generate_graphs(ns: List[int], ks: List[float]):
    for n in ns:
        generate_graph(n, ks)
