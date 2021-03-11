from conjugate_gradients import conjugate_gradients
from forms.quadratic import generate_random_form
from graphs.trajectories import draw_all
from newton import newton
from optimize.unidimensional.fibonacci import FibonacciOptimizer
from optimize.optimizer import Optimizer
from common.oracle import Oracle
from gradient import gradient_descent, linear_search
from graphs import condition_number, trajectories

import numpy as np

if __name__ == '__main__':
    f = Oracle(2, lambda x, y: (x - 1) ** 2 + (y + 1) ** 2, lambda x, y: np.array([2 * x - 2, 2 * y + 2]),
               lambda x, y: [[2.0, 0.0], [0.0, 2.0]])
    descent_result = gradient_descent(f, np.array([-1.0, 1.0]), lambda g: FibonacciOptimizer(g, (0.0, 1.0), 0.001),
                                      dx=1e-5, df=1e-6)
    print(f"optimal point: {descent_result.optimal_point}")
    conj_result = conjugate_gradients(f, np.array([-1.0, 1.0]), lambda g: FibonacciOptimizer(g, (0.0, 1.0), 0.001),
                                      dx=1e-5, df=1e-6)
    print(f"other point: {conj_result.optimal_point}")
    newton_result = newton(f, np.array([-1.0, 1.0]), lambda g: FibonacciOptimizer(g, (0.0, 1.0), 0.001),
                                      dx=1e-5, df=1e-6)
    print(f"newton point: {newton_result.optimal_point}")

    # draw_all()
    # print(linear_search(f, (np.array([-2, -2]), np.array([2, 2])), 0.01))
    # condition_number.generate_graphs([2, 3, 5, 10, 20, 30, 40, 50],
    #                                  [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0])
