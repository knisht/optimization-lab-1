from typing import List

import numpy as np

from common.oracle import Oracle


class OptimizationResult:
    def __init__(self, initial_point: np.ndarray, optimal_point: np.ndarray, iterations: int,
                 trajectory: List[np.ndarray], cause: str, oracle: Oracle, name: str):
        self.initial_point = initial_point
        self.optimal_point = optimal_point
        self.iterations = iterations
        self.trajectory = trajectory
        self.cause = cause
        self.oracle = oracle
        self.name = name

    def print_info(self, print_trajectory: bool = False):
        print(f"""
        {self.name} on {self.initial_point}:
        Cause: {self.cause}
        Optimal point: {self.optimal_point}
        Function value at point: {self.oracle.f(*self.optimal_point)}
        Iterations passed: {self.iterations} 
        {f"Trajectory: {self.trajectory}" if print_trajectory else ""}""")
