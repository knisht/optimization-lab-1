from math import exp
from typing import Callable

import numpy as np

from graphs.trajectories2 import plot_trajectory
from optimize.multidimensional.OptimizationResult import OptimizationResult
from common.oracle import Oracle
from optimize.unidimensional.fibonacci import FibonacciOptimizer
from optimize.multidimensional.ConjugateGradients import ConjugateGradients
from optimize.multidimensional.GradientDescent import GradientDescent
from optimize.multidimensional.Newton import Newton
from optimize.unidimensional.golden_ratio import GoldenRatioOptimizer

def print_value(x):
    print("{0:0.3f}".format(x), end="")

def print_point(x, y):
    print("(", end='')
    print_value(x)
    print(", ", end='')
    print_value(y)
    print(")", end=' ')

if __name__ == '__main__':
    functions = [lambda x, y: 100 * (y - x) ** 2 + (1 - x) ** 2,
                 lambda x, y: 100 * (y - x ** 2) ** 2 + (1 - x) ** 2,
                 lambda x, y: (-1) * (2 * exp(-((x - 1) / 2) ** 2 - (y - 1) ** 2) +
                                      3 * exp(-((x - 2) / 3) ** 2 - ((y - 3) / 2) ** 2))]
    gradients = [lambda x, y: np.array([-200 * (y - x) + 2 * x - 2, 200 * (y - x)]),
                 lambda x, y: np.array([-400 * (y - x ** 2) * x - 2 * (1 - x), 200 * (y - x ** 2)]),
                 lambda x, y: (-1.0) * np.array([2 * exp(-((x - 1) / 2) ** 2 - (y - 1) ** 2) * (-(x - 1) / 2) +
                                                 3 * exp(-((x - 2) / 3) ** 2 - ((y - 3) / 2) ** 2) * (2.0 / 3.0) * (
                                                             -(x - 2) / 3),
                                                 2 * exp(-((x - 1) / 2) ** 2 - (y - 1) ** 2) * (-2 * (y - 1)) +
                                                 3 * exp(-((x - 2) / 3) ** 2 - ((y - 3) / 2) ** 2) * (-((y - 3) / 2))])]
    hesse = [lambda x, y: np.array([[202, -200], [-200, 200]]),
             lambda x, y: np.array([[-400 * ((y - x ** 2) - 2 * x ** 2) + 2, -400 * x], [-400 * x, 200]]),
             lambda x, y: (-1) * np.array(
                 [[3 * ((4 / 81) * (x - 2) ** 2 * exp((-1 / 9) * (x - 2) ** 2 - (1 / 4) * (y - 3) ** 2) -
                        (2 / 9) * exp((-1 / 9) * (x - 2) ** 2 - (1 / 4) * (y - 3) ** 2)) +
                   2 * ((1 / 4) * (x - 1) ** 2 * exp((-(1 / 4) * (1 - x) ** 2 - (y - 1) ** 2)) -
                        (1 / 2) * exp(-(1 / 4) * (1 - x) ** 2 - (y - 1) ** 2)),
                   (1 / 3) * (x - 2) * (y - 3) * exp(-(1 / 9) * (x - 2) ** 2 - (1 / 4) * (y - 3) ** 2) +
                   2 * (x - 1) * (y - 1) * exp(-(1 / 4) * (x - 1) ** 2 - (y - 1) ** 2)],
                  [(1 / 3) * (x - 2) * (y - 3) * exp(-(1 / 9) * (x - 2) ** 2 - (1 / 4) * (y - 3) ** 2) +
                   2 * (x - 1) * (y - 1) * exp(-(1 / 4) * (x - 1) ** 2 - (y - 1) ** 2),
                   3 * ((1 / 4) * (y - 3) ** 2 * exp((-1 / 9) * (x - 2) ** 2 - (1 / 4) * (3 - y) ** 2) -
                        (1 / 2) * exp((-1 / 9) * (x - 2) ** 2 - (1 / 4) * (3 - y) ** 2)) +
                   2 * (4 * (y - 1) ** 2 * exp((-1 / 4) * (x - 1) ** 2 - (y - 1) ** 2) -
                        2 * exp((-1 / 4) * (x - 1) ** 2 - (y - 1) ** 2))]])]
    reprs = ["100(y-x)^2 + (1-x)^2", "100(y-x^2)^2 + (1-x)^2",
             "2exp(-((x-1)/2)^2 - (y-1)^2)+3exp(-((x-2)/3)^2-((y-3)/2)^2)"]

    initial_points = [
        np.array([0, 1]),
        np.array([2, 2]),
        np.array([-1.0, 1.0]),
        np.array([3.0, 2.0]),
        np.array([0.1, -0.5]),
    ]
    for x0 in initial_points:
        for i in range(3):
            descent_results = []
            conjugate_results = []
            newton_results = []
            f = Oracle(2, functions[i], gradients[i], hesse[i], reprs[i])
            descent_result = GradientDescent().run(f, x0,
                                                   lambda g: GoldenRatioOptimizer(g, (0.0, 100.01), 1e-5),
                                                   df=1e-6, iterations=50)
            descent_results.append(descent_result)
            conj_result = ConjugateGradients().run(f, x0,
                                                   lambda g: FibonacciOptimizer(g, (0.0, 1.01), 1e-7),
                                                   df=1e-6, iterations=50)
            conjugate_results.append(conj_result)
            newton_result = Newton().run(f, x0, lambda g: FibonacciOptimizer(g, (0.0, 1.01), 1e-5), df=1e-7,
                                         iterations=50)
            newton_results.append(newton_result)
            plot_trajectory([descent_result, conj_result, newton_result])

            plot_trajectory(descent_results)
            plot_trajectory(conjugate_results)
            descent_result.print_info()
            conj_result.print_info()
            newton_result.print_info()

            # print_point(*newton_result.trajectory[0])
            # print_point(*newton_result.trajectory[-1])
            # print(newton_result.iterations, end=' ')
            # print_value(f.f(*newton_result.trajectory[-1]))
            print("====================")
        # plot_trajectory(newton_results)
