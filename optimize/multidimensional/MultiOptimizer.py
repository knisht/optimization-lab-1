import time
import tracemalloc
from typing import Callable, Optional, Any, Tuple

import numpy as np

from optimize.multidimensional.OptimizationResult import OptimizationResult
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

    def name(self) -> str:
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
        memory_consumption = 0
        elapsed_time = 0.0
        while (iterations is None) or (iteration_count < iterations):
            tracemalloc.start()
            start_timestamp = time.time()
            x1, payload = self.iteration(f, x, step_optimizer, iteration_count, payload)
            end_timestamp = time.time()
            res = tracemalloc.take_snapshot()
            elapsed_time += end_timestamp - start_timestamp
            stats = res.statistics(cumulative=True, key_type='filename')
            for stat in stats:
                memory_consumption += stat.size
            tracemalloc.stop()
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
            if np.linalg.norm(f.grad(*x)) < 1e-10:
                cause = "Plateau"
                break
            iteration_count += 1
            x = x1
        if cause is None:
            cause = "Exceed limit of iterations"
        return OptimizationResult(x0, x, iteration_count, trajectory, cause, f, self.name(), memory_consumption, elapsed_time)
