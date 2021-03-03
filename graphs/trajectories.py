from typing import Callable, Tuple, List

import numpy as np
from matplotlib import pyplot as plt

from common.oracle import Oracle
from gradient import gradient_descent
from optimize.optimizer import Optimizer
from optimize.methods.fibonacci import FibonacciOptimizer
from optimize.methods.bisection import BisectionOptimizer
from optimize.methods.golden_ratio import GoldenRatioOptimizer


def compute_trajectory(f: Callable[[float, float], float], jacobi: Callable[[float, float], np.ndarray],
                       optimizer: Callable[[Callable], Optimizer]) -> Tuple[np.ndarray, List[np.ndarray]]:
    oracle = Oracle(2, f, jacobi)
    initial_point = np.array([4.0, 4.0])
    argmin, _, trajectory = gradient_descent(f=oracle, x0=initial_point, step_optimizer=optimizer, df=1e-7,
                                             iterations=100, dx=1e-7)
    return argmin, trajectory


def generate_graph(f: Callable[[float, float], float], jacobi: Callable[[float, float], np.ndarray],
                   representation: str,
                   func_name: str,
                   xl: float, xr: float, yl: float, yr: float):
    def ternary_search(g):
        return Optimizer(g, (0, 2), 0.001)

    def binary_search(g):
        return BisectionOptimizer(g, (0, 2), 0.001)

    def golden_ratio(g):
        return GoldenRatioOptimizer(g, (0, 2), 0.001)

    optimizers = [ternary_search, golden_ratio, binary_search]
    trajectory_names = ['Ternary search', 'Golden ratio', 'Binary search']
    descent_results = list(map(lambda op: compute_trajectory(f, jacobi, op), optimizers))
    argmin = descent_results[0][0]

    v_func = np.vectorize(lambda x, y: f(x, y))
    xx, yy = np.meshgrid(np.linspace(xl, xr, 100),
                         np.linspace(yl, yr, 100))

    fig, ax = plt.subplots()
    qx = ax.contour(xx, yy, v_func(xx, yy), [f(argmin[0], argmin[1]) + i for i in range(-5, 5)],
                    linestyles=('solid'))
    for (_, trajectory), trajectory_name in zip(descent_results, trajectory_names):
        __plot_trajectory(ax, trajectory, trajectory_name)
    ax.clabel(qx, fontsize=9, fmt='%.1f', inline=1)
    ax.legend()
    ax.set_title(func_name)
    plt.savefig("results/" + representation + ".png")
    plt.show()


def __plot_trajectory(ax, trajectory: List[np.ndarray], name: str):
    x_trajectories = list(map(lambda t: t[0], trajectory))
    y_trajectories = list(map(lambda t: t[1], trajectory))
    ax.plot(x_trajectories, y_trajectories, label=name)


def draw_all():
    functions = [lambda x, y: x * x - 3 * x * y + 5 * y * y,
                 lambda x, y: 4 * x * x + 20 * y * y,
                 lambda x, y: x * x + x * y + 3 * y * y]
    jacobies = [lambda x, y: np.array([2 * x - 3 * y, -3 * x + 10 * y]),
                lambda x, y: np.array([8 * x, 40 * y]),
                lambda x, y: np.array([2 * x + y, 6 * y + x])]
    representations = ["func_1", "func_2", "func_3"]
    function_names = ["x^2 - 3xy + 5y^2", "4x^2 + 20y^2", "x^2 + xy + 3y^2"]
    ranges = [(-3.0, 4.5, -3.0, 3.0), (-5, 5, -0.5, 0.5), (-4.0, 4.0, -2.0, 2.0)]
    initial_points = [[], [], []]
    for f, jacobi, repr, f_name, (xl, xr, yl, yr), points in zip(functions, jacobies, representations, function_names, ranges, initial_points):
        generate_graph(f, jacobi, repr, f_name, xl, xr, yl, yr, points)
