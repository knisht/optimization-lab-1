from typing import Callable, Union, List


class Oracle:
    """
    :param n: function input dimension
    :param whitebox: explicit gradient calculator
        example. f(x1, x2) = x1^2 - x1 * x2,
                 whitebox(x1, x2) = [2x1 - x2, -x1]
    """

    def __init__(
            self, n: int, f: Callable[..., float],
            whitebox: Union[None, Callable[..., List[float]]]
    ):
        self.n = n
        self.f = f
        self.whitebox = whitebox

    def __call__(self, *args):
        assert len(args) == self.n
        return self.f(*args)

    def grad(self, *args, step=None):
        assert len(args) == self.n
        if self.whitebox is not None:
            return self.whitebox(*args)

        assert step is not None and step != 0.0
        grad = []
        value = self.f(*args)
        for i in range(n):
            args[i] += step
            grad.append((f(args) - value) / step)
            args[i] -= step

        return grad
