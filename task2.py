from math import exp

import numpy as np

from OptimizationResult import OptimizationResult
from common.oracle import Oracle
from optimize.methods.fibonacci import FibonacciOptimizer

from optimize.multidimensional.ConjugateGradients import ConjugateGradients
from optimize.multidimensional.GradientDescent import GradientDescent
from optimize.multidimensional.Newton import Newton


def print_info(result: OptimizationResult, name: str, functions: list):
    print(f"""
{name}:
Cause: {result.cause}
Optimal point: {result.optimal_point}
Function value at point: {functions[i](*result.optimal_point)}
Iteration: {result.iterations}""")


if __name__ == '__main__':
    functions = [lambda x, y: 100 * (y - x) ** 2 + (1 - x) ** 2,
                 lambda x, y: 100 * (y - x ** 2) ** 2 + (1 - x) ** 2,
                 lambda x, y: 2 * exp(-((x - 1) / 2) ** 2 - (y - 1) ** 2) +
                              3 * exp(-((x - 2) / 3) ** 2 - ((y - 3) / 2) ** 2)]
    gradients = [lambda x, y: np.array([-200 * (y - x) - x, 200 * (y - x)]),
                 lambda x, y: np.array([-400 * (y - x ** 2) * x - 2 * (1 - x), 200 * (y - x ** 2)]),
                 lambda x, y: np.array([2 * exp(-((x - 1) / 2) ** 2 - (y - 1) ** 2) * (-(x - 1) / 2) +
                                        3 * exp(-((x - 2) / 3) ** 2 - ((y - 3) / 2) ** 2) * (2.0 / 3.0) * (
                                                -(x - 2) / 3),
                                        2 * exp(-((x - 1) / 2) ** 2 - (y - 1) ** 2) * (-2 * (y - 1)) +
                                        3 * exp(-((x - 2) / 3) ** 2 - ((y - 3) / 2) ** 2) * (-((y - 3) / 2))])]
    hesse = [lambda x, y: [[199, -200], [-200, 200]],
             lambda x, y: [[-400 * ((y - x ** 2) - 2 * x ** 2) + 2, -400 * x], [-400 * x, 200]],
             lambda x, y: [[3 * ((4 / 81) * (x - 2) ** 2 * exp((-1 / 9) * (x - 2) ** 2 - (1 / 4) * (y - 3) ** 2) -
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
                                 2 * exp((-1 / 4) * (x - 1) ** 2 - (y - 1) ** 2))]]]

    initial_points = [np.array([-100.0, 100.0]), np.array([11.0, 2.0]), np.array([-5.0, -0.5]), np.array([0.1, -10.1]),
                      np.array([0.1, 0.1])]
    for x0 in initial_points:
        # for i in range(3):
        i = 1
        print(i)
        print(x0)
        f = Oracle(2, functions[i], gradients[i], hesse[i])
        descent_result = GradientDescent().run(f, x0,
                                               lambda g: FibonacciOptimizer(g, (0.0, 1.01), 1e-5),
                                               dx=1e-5, df=1e-6, iterations=2000)
        print_info(descent_result, "Gradient descent", functions)
        # print(f"trajectory: {descent_result.trajectory}")
        conj_result = ConjugateGradients().run(f, x0,
                                               lambda g: FibonacciOptimizer(g, (0.0, 1.01), 1e-5),
                                               dx=1e-5, df=1e-6, iterations=2000)
        print_info(conj_result, "Conjugate gradients", functions)
        # print(f"trajectory: {conj_result.trajectory}")

        newton_result = Newton().run(f, x0, lambda g: FibonacciOptimizer(g, (0.0, 1.01), 1e-5),
                                     dx=1e-5, df=1e-6, iterations=2000)
        print_info(newton_result, "Newton", functions)
        # print("======================")
