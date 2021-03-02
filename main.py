from forms.quadratic import generate_random_form
from graphs.trajectories import draw_all
from optimize.optimizer import Optimizer
from common.oracle import Oracle
from gradient import gradient_descent
from graphs import condition_number, trajectories

import numpy as np

if __name__ == '__main__':
    draw_all()
    # condition_number.generate_graphs([2, 3, 5, 10, 20, 30, 40, 50],
    #                                  [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0])
