from typing import Callable, Tuple, List
import numpy as np
from common.oracle import Oracle
from gradient import gradient_descent
from optimize.methods.constant import ConstantOptimizer
from optimize.optimizer import Optimizer
from optimize.methods.fibonacci import FibonacciOptimizer
from optimize.methods.bisection import BisectionOptimizer
from optimize.methods.golden_ratio import GoldenRatioOptimizer
from matplotlib import pyplot as plt


def compute_trajectory(
        f: Callable[[float, float], float], jacobi: Callable[[float, float], np.ndarray],
        optimizer: Callable[[Callable], Optimizer], initial_points: List[np.ndarray]
) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    oracle = Oracle(2, f, jacobi)
    trajectories = []
    argmin, _, _ = gradient_descent(
        f=oracle, x0=initial_points[0], step_optimizer=optimizer,
        df=1e-7, iterations=100, dx=1e-7
    )

    for point in initial_points:
        _, _, trajectory = gradient_descent(
            f=oracle, x0=point, step_optimizer=optimizer,
            df=1e-7, iterations=100, dx=1e-7
        )
        trajectories.append(trajectory)
    return argmin, trajectories


def compute_iterations(
        f: Callable[[float, float], float], jacobi: Callable[[float, float], np.ndarray],
        optimizer: Callable[[Callable], Optimizer], initial_points: List[np.ndarray], power: int = 10
):
    oracle = Oracle(2, f, jacobi)
    iterations = np.zeros((power + 1,))

    for point in initial_points:
        x, eps = point, 1.0
        for d in range(1, power + 1):
            x, it, _ = gradient_descent(f=oracle, x0=x, step_optimizer=optimizer, df=eps, dx=eps)
            iterations[d] += it
            eps /= 10.0

    iterations = np.cumsum(iterations)
    return iterations / len(initial_points)


def __build_optimizer(op):
    return lambda g: op(g, (0, 0.5), 0.05)


optimizers = list(map(__build_optimizer, [Optimizer, BisectionOptimizer, GoldenRatioOptimizer, FibonacciOptimizer,
                                          lambda g, b, eps: ConstantOptimizer(g, b, eps, 0.05)]))
optimizer_names = ['Ternary search', 'Golden ratio', 'Binary search', 'Fibonacci', 'Constant']
optimizer_colors = ['r', 'b', 'g', 'k', 'm']


def __plot_trajectory(ax, trajectories: List[List[np.ndarray]], name: str, color: str):
    ok = False
    for trajectory in trajectories:
        x_trajectories = list(map(lambda t: t[0], trajectory))
        y_trajectories = list(map(lambda t: t[1], trajectory))
        if not ok:
            ax.plot(x_trajectories, y_trajectories, color, marker=',', linewidth=1, markersize=1, label=name)
            ok = True
        else:
            ax.plot(x_trajectories, y_trajectories, color, marker=',', linewidth=1, markersize=1)


def generate_trajectories_graph(
        f: Callable[[float, float], float], jacobi: Callable[[float, float], np.ndarray],
        representation: str, func_name: str,
        xl: float, xr: float, yl: float, yr: float, points: List[np.ndarray],
        level_lines
):
    descent_results = list(map(lambda op: compute_trajectory(f, jacobi, op, points), optimizers))
    v_func = np.vectorize(lambda x, y: f(x, y))
    xx, yy = np.meshgrid(np.linspace(xl, xr, 100),
                         np.linspace(yl, yr, 100))

    fig, ax = plt.subplots()
    qx = ax.contour(xx, yy, v_func(xx, yy), level_lines,
                    linestyles=('solid'))
    for (_, trajectories), trajectory_name, color in zip(descent_results, optimizer_names, optimizer_colors):
        __plot_trajectory(ax, trajectories, trajectory_name, color)
    ax.clabel(qx, fontsize=5, fmt='%.2f', inline=1)
    ax.legend()
    ax.set_title(func_name)
    plt.savefig("results/" + representation + ".png")
    plt.show()


def generate_iterations_graph(
        f: Callable[[float, float], float], jacobi: Callable[[float, float], np.ndarray],
        representation: str, func_name: str, power: int, points: List[np.ndarray]
):
    descent_results = list(map(lambda op: compute_iterations(f, jacobi, op, points, power), optimizers))
    fig, ax = plt.subplots()
    ax.set_xlabel('-log_10(eps)')
    for iterations, optimizer_name, color in zip(descent_results, optimizer_names, optimizer_colors):
        ax.plot(range(power + 1), iterations, color, label=optimizer_name)
    ax.legend()
    ax.set_title(func_name)
    plt.savefig("results/" + representation + ".png")
    plt.show()


def generate_bounds_graph(f: Callable[[float, float], float], fn_name: str, l, r, esps: List[float]):
    for e in esps:
        ops = list(map(lambda op: op(f, (l, r), e), [BisectionOptimizer, GoldenRatioOptimizer, FibonacciOptimizer]))
        results = []
        for op in ops:
           results.append(op.optimize())
        fig, ax = plt.subplots()
        #ax.set_xlabel('-log_10(eps)')
        names = ['BisectionOptimizer', 'GoldenRatioOptimizer', 'FibonacciOptimizer']
        colors = ['r', 'b', 'y']
        for op, optimizer_name, color in zip(ops,names, colors):
            a = [l] + [it[0] for it in op.history]
            b = [r] + [it[1] for it in op.history]
            iters = list(range(len(b)))
            ax.plot(iters, a, color, label=f"{optimizer_name};  f_calls_cnt: {op.f_calls}")
            ax.plot(iters, b, color)
        ax.legend()
        ax.set_title(f"f = {fn_name}, e = {e}")
        plt.savefig("results/" + f'1-{e}' + ".png")
        plt.show()


def draw_all():


    # 1
    fn = lambda x: x ** 3  - 3 * (x **2) - 4 * x + 10
    generate_bounds_graph(fn, "x^3 - 3x^2 - 4x + 1", 2, 5, [1, 0.1, 0.01, 0.001, 0.000001])


    functions = [lambda x, y: x * x - 3 * x * y + 5 * y * y,
                 lambda x, y: 4 * x * x + 20 * y * y,
                 lambda x, y: x * x + x * y + 3 * y * y]
    jacobies = [lambda x, y: np.array([2 * x - 3 * y, -3 * x + 10 * y]),
                lambda x, y: np.array([8 * x, 40 * y]),
                lambda x, y: np.array([2 * x + y, 6 * y + x])]
    representations = ["func_1", "func_2", "func_3"]
    function_names = ["x^2 - 3xy + 5y^2", "4x^2 + 20y^2", "x^2 + xy + 3y^2"]
    ranges = [(-1.0, 1.0, -0.4, 0.6), (-0.75, 0.75, -0.3, 0.3), (-1.2, 1.6, -1.0, 1.0)]
    initial_points = [[np.array([-0.2, -0.25]), np.array([0.25, -0.3])],
                      [np.array([-0.5, -0.1]), np.array([0.5, 0.1])],
                      [np.array([1.5, 0.5]), np.array([-1.0, 1.0])]]
    level_lines = [[float(i + 1) / 20.0 for i in range(10)],
                   [float(i + 1) / 10.0 for i in range(10)],
                   [float(i + 1) / 7.0 for i in range(10)]]
    for f, jacobi, repr, f_name, (xl, xr, yl, yr), points, level_line in zip(functions, jacobies, representations,
                                                                             function_names, ranges, initial_points,
                                                                             level_lines):
        generate_trajectories_graph(f, jacobi, repr, f_name, xl, xr, yl, yr, points, level_line)
        generate_iterations_graph(f, jacobi, repr + "_iter", f_name, 10, points)
