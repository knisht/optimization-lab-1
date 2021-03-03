from typing import Tuple, Callable

from optimize.optimizer import Optimizer


class BisectionOptimizer(Optimizer):
    '''
    TODO сорри, просто 1. я не понял зачем тут такой if, 2. поменял сигнатуру степа (написал внизу как мне кажется ок)
    def _step(self, a: float, b: float) -> Tuple[float, float]:
        if self.f(a) * self.f(b) > 0:
            raise Exception("No root found")
        else:
            mid = (a + b) / 2.0
            if self.f(a) * self.f(mid) < 0:
                b = mid
            else:
                a = mid
        return a, b
    '''

    def _step(self, a: float, b: float) -> Tuple[float, float]:
        mid = (a + b) / 2.0
        return mid - self.eps / 3, mid + self.eps / 3
