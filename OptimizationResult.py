from typing import List

import numpy as np


class OptimizationResult:
    def __init__(self, optimal_point: np.ndarray, iterations: int, trajectory: List[np.ndarray], cause: str):
        self.optimal_point = optimal_point
        self.iterations = iterations
        self.trajectory = trajectory
        self.cause = cause

