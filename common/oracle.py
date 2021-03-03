from typing import Callable, List, Optional

import numpy as np


class Oracle:
    def __init__(
            self, n: int, f: Callable[..., float],
            whitebox: Optional[Callable[..., np.ndarray]]
    ):
        """
        :param n: function input dimension
        :param whitebox: explicit gradient calculator
            example. f(x1, x2) = x1^2 - x1 * x2,
                     whitebox(x1, x2) = [2x1 - x2, -x1]
        """
        self.n = n
        self.f = f
        self.whitebox = whitebox
        self.calls = 0

    def __call__(self, *args):
        assert len(args) == self.n
        self.calls += 1
        return self.f(*args)

    def reset(self):
        self.calls = 0

    def grad(self, *args, step=None) -> np.ndarray:
        assert len(args) == self.n
        if self.whitebox is not None:
            return self.whitebox(*args)

        assert step is not None and step != 0.0
        grad = np.zeros((self.n,))
        value = self.f(*args)
        for i in range(self.n):
            args[i] += step
            grad[i] = (self.f(*args) - value) / step
            args[i] -= step

        return grad

    def stats(self):
        return {
            'calls': self.calls
        }