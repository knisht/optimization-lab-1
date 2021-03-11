from typing import Callable, Optional, Any, Tuple

import numpy as np

from OptimizationResult import OptimizationResult
from common.oracle import Oracle
from optimize.optimizer import Optimizer


class MultiOptimizer:

    def init(self) -> Any:
        pass

    def iteration(self, f: Oracle, x: np.ndarray,
                  optimizer: Callable[[Callable], Optimizer],
                  iteration: int,
                  payload: Any) -> Tuple[np.ndarray, Any]:
        pass

    def run(self, f: Oracle, x0: np.ndarray,
            step_optimizer: Callable[[Callable], Optimizer],
            iterations: Optional[int] = None,
            dx: Optional[float] = None,
            df: Optional[float] = None) -> OptimizationResult:
        x = x0
        iteration_count = 0
        trajectory = [x]
        cause = None
        payload = self.init()
        while (iterations is None) or (iteration_count < iterations):
            x1, payload = self.iteration(f, x, step_optimizer, iteration_count, payload)
            trajectory.append(x1)
            delta = np.linalg.norm(x1 - x)
            if dx is not None and np.linalg.norm(delta) < dx:
                cause = "Exceed limit of accuracy for argument"
                break
            if df is not None and abs(f(*x1) - f(*x)) < df:
                cause = "Exceed limit of accuracy for function"
                break
            if x.dot(x) > 1e10:
                cause = "Divergence"
                break
            iteration_count += 1
            x = x1

        if cause is None:
            cause = "Exceed limit of iterations"
        return OptimizationResult(x, iteration_count, trajectory, cause)
