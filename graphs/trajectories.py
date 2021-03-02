from typing import Callable
import numpy as np
from common.oracle import Oracle
from forms.quadratic import generate_random_form
from gradient import gradient_descent
from optimize.optimizer import Optimizer
from matplotlib import pyplot as plt


def generate_graph(f: Callable[[float, float], float], jacobi: Callable[[float, float], np.ndarray],
                   representation: str,
                   func_name: str,
                   xl: float,
                   xr: float,
                   yl: float,
                   yr: float):
    oracle = Oracle(2, f, jacobi)

    def ternary_search(g):
        return Optimizer(g, (0, 2), 0.001)

    initial_point = np.array([4.0, 4.0])
    argmin, _, trajectory = gradient_descent(f=oracle, x0=initial_point, step_optimizer=ternary_search, df=1e-7,
                                             iterations=100, dx=1e-7)
    x_trajectories = list(map(lambda t: t[0], trajectory))
    y_trajectories = [trajectory[i][1] for i in range(len(trajectory))]
    print(x_trajectories)
    print(y_trajectories)
    v_func = np.vectorize(lambda x, y: f(x, y))
    xx, yy = np.meshgrid(np.linspace(xl, xr, 100),
                         np.linspace(yl, yr, 100))

    fig, ax = plt.subplots()
    qx = ax.contour(xx, yy, v_func(xx, yy), [f(argmin[0], argmin[1]) + i for i in range(-5, 5)],
                linestyles=('solid'))
    ax.plot(x_trajectories, y_trajectories)
    ax.clabel(qx, fontsize=9, fmt='%.1f', inline=1)
    ax.set_title(func_name)
    plt.savefig("results/" + representation + ".png")
    plt.show()

def draw_all():
    functions = [lambda x, y: x*x - 3 * x * y + 5 * y * y, lambda x, y: 4*x*x + 20 * y * y, lambda x, y: x * x + x * y + 3 * y * y]
    jacobies = [lambda x, y: np.array([2*x - 3 * y, -3 * x + 10 * y]), lambda x, y: np.array([8 * x, 40 * y]),
                lambda x, y: np.array([2 * x + y, 6 * y + x])]
    representations = ["func_1", "func_2", "func_3"]
    function_names = ["xy", "x^2 + y^2", "x^2 + xy + 3y^2"]
    ranges = [(-3.0, 4.5, -3.0, 3.0), (-5, 5, -0.5, 0.5), (-4.0, 4.0, -2.0, 2.0)]
    for f, jacobi, repr, f_name, (xl, xr, yl, yr) in zip(functions, jacobies, representations, function_names, ranges):
        generate_graph(f, jacobi, repr, f_name, xl, xr, yl, yr)
