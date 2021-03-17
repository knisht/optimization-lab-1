import time
import tracemalloc
from typing import Callable, Optional, Any, Tuple

import numpy as np

from optimize.multidimensional.optimization_result import OptimizationResult
from common.oracle import Oracle
from optimize.unidimensional.optimizer import Optimizer
from common.stats import StatCollector


class MultiOptimizer:

    def __init__(self):
        self._stats = StatCollector()

    def init(self) -> Any:
        pass

    def iteration(
            self, f: Oracle, x: np.ndarray,
            optimizer: Callable[[Callable], Optimizer],
            iteration: int,
            payload: Any
    ) -> Tuple[np.ndarray, Any]:
        pass

    def name(self) -> str:
        pass

    def run(
            self, f: Oracle, x0: np.ndarray,
            step_optimizer: Callable[[Callable], Optimizer],
            iterations: Optional[int] = None,
            dx: Optional[float] = None,
            df: Optional[float] = None
    ) -> OptimizationResult:
        x = x0
        iteration_count = 0
        trajectory = [x]
        cause = None
        payload = self.init()
        self._stats.clear()
        f.reset_stat()

        while (iterations is None) or (iteration_count < iterations):
            self._stats.start()
            x1, payload = self.iteration(f, x, step_optimizer, iteration_count, payload)
            self._stats.stop()

            trajectory.append(x1)
            delta = np.linalg.norm(x1 - x)
            self._stats.report('*', x.size)
            self._stats.report('+', x.size - 1)
            # self._stats.report('sqrt', 1)

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

        self._stats.assimilate(f)
        if cause is None:
            cause = "Exceed limit of iterations"

        arithm_stats = self._stats.arithm
        if len(self._stats.extra) > 0:
            arithm_stats = f'{arithm_stats} (+{",".join(list(map(lambda x: str(x[1] * 15), self._stats.extra.items())))})'
        calls = [self._stats.fcalls, self._stats.gcalls, self._stats.hcalls]

        return OptimizationResult(
            x0, x, iteration_count, trajectory, cause, f, self.name(),
            self._stats.memory, self._stats.elapsed_time, arithm_stats, calls
        )
