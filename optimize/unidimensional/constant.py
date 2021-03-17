from typing import Callable, Tuple

from optimize.unidimensional.optimizer import Optimizer


class ConstantOptimizer(Optimizer):
    def __init__(self, f: Callable[[float], float], bounds: Tuple[float, float], eps: float, step_eps: float):
        super().__init__(f, bounds, eps)
        self.step_eps = step_eps

    def optimize(self) -> float:
        return self.step_eps