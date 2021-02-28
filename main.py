from utils.optimizer import Optimizer
from utils.oracle import Oracle

from gradient import gradient_descent

import numpy as np


if __name__ == '__main__':
    f = Oracle(
        2, lambda x, y: (x - 2) ** 2 + (y - 3) ** 2 + x * y,
        lambda x, y: np.array([2 * x - 4 + y, 2 * y - 6 + x])
    )
    ternary_search = Optimizer

    x = gradient_descent(
        f=f, x0=np.array([-6.0, 17.0]), step_optimizer=ternary_search
    )
    print(x, f(*x))

