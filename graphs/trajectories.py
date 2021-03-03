from typing import Callable, Tuple, List
import numpy as np
from common.oracle import Oracle
from gradient import gradient_descent
from optimize.optimizer import Optimizer
from optimize.methods.fibonacci import FibonacciOptimizer
from optimize.methods.bisection import BisectionOptimizer
from optimize.methods.golden_ratio import GoldenRatioOptimizer
from matplotlib import pyplot as plt


def compute_trajectory(f: Callable[[float, float], float], jacobi: Callable[[float, float], np.ndarray],
                       optimizer: Callable[[Callable], Optimizer], initial_points: List[np.ndarray]) -> Tuple[np.ndarray, list[list[np.ndarray]]]:
    oracle = Oracle(2, f, jacobi)
    trajectories = []
    argmin, _, _ = gradient_descent(f=oracle, x0=initial_points[0], step_optimizer=optimizer, df=1e-7,
                                             iterations=100, dx=1e-7)
    for point in initial_points:
        _, _, trajectory = gradient_descent(f=oracle, x0=point, step_optimizer=optimizer, df=1e-7, iterations=100, dx=1e-7)
        trajectories.append(trajectory)
    return argmin, trajectories


def generate_graph(f: Callable[[float, float], float], jacobi: Callable[[float, float], np.ndarray],
                   representation: str,
                   func_name: str,
                   xl: float, xr: float, yl: float, yr: float, points: List[np.ndarray]):
    def build_optimizer(op):
        return lambda g: op(g, (0, 0.5), 0.01)

    optimizers = list(map(build_optimizer, [Optimizer, BisectionOptimizer, GoldenRatioOptimizer]))
    trajectory_names = ['Ternary search', 'Golden ratio', 'Binary search']
    trajectory_colors = ['r', 'b', 'g']
    descent_results = list(map(lambda op: compute_trajectory(f, jacobi, op, points), optimizers))
    argmin = descent_results[0][0]

    v_func = np.vectorize(lambda x, y: f(x, y))
    xx, yy = np.meshgrid(np.linspace(xl, xr, 100),
                         np.linspace(yl, yr, 100))

    fig, ax = plt.subplots()
    qx = ax.contour(xx, yy, v_func(xx, yy), [f(argmin[0], argmin[1]) + float(i)/3.0 for i in range(-10, 10)],
                    linestyles=('solid'))
    for (_, trajectories), trajectory_name, color in zip(descent_results, trajectory_names, trajectory_colors):
        __plot_trajectory(ax, trajectories, trajectory_name, color)
    ax.clabel(qx, fontsize=5, fmt='%.1f', inline=1)
    ax.legend()
    ax.set_title(func_name)
    plt.savefig("results/" + representation + ".png")
    plt.show()


def __plot_trajectory(ax, trajectories: List[List[np.ndarray]], name: str, color: str):
    ok = False
    for trajectory in trajectories:
        x_trajectories = list(map(lambda t: t[0], trajectory))
        y_trajectories = list(map(lambda t: t[1], trajectory))
        if not ok:
            ax.plot(x_trajectories, y_trajectories, color, marker=',', linewidth=1, markersize=1, label=name)
            ok = True
        else:
            ax.plot(x_trajectories, y_trajectories, color, marker=',',  linewidth=1, markersize=1)


def draw_all():
    functions = [lambda x, y: x * x - 3 * x * y + 5 * y * y,
                 lambda x, y: 4 * x * x + 20 * y * y,
                 lambda x, y: x * x + x * y + 3 * y * y]
    jacobies = [lambda x, y: np.array([2 * x - 3 * y, -3 * x + 10 * y]),
                lambda x, y: np.array([8 * x, 40 * y]),
                lambda x, y: np.array([2 * x + y, 6 * y + x])]
    representations = ["func_1", "func_2", "func_3"]
    function_names = ["x^2 - 3xy + 5y^2", "4x^2 + 20y^2", "x^2 + xy + 3y^2"]
    ranges = [(-2.0, 2.0, -1.5, 1.5), (-2.3, 3.0, -0.5, 1.0), (-2.5, 2.5, -1.5, 2.0)]
    initial_points = [[np.array([-0.5, 0.5]), np.array([1.0, -0.5])],
                      [np.array([2.0, 0.5]), np.array([-2.0, 0.5])],
                      [np.array([2.0, 1.5]), np.array([-1.0, 1.0])]]
    for f, jacobi, repr, f_name, (xl, xr, yl, yr), points in zip(functions, jacobies, representations, function_names, ranges, initial_points):
        generate_graph(f, jacobi, repr, f_name, xl, xr, yl, yr, points)
