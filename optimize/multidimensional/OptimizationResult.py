from typing import List

import numpy as np

from common.oracle import Oracle


class OptimizationResult:
    def __init__(self, initial_point: np.ndarray, optimal_point: np.ndarray, iterations: int,
                 trajectory: List[np.ndarray], cause: str, oracle: Oracle,
                 name: str,
                 memory_consumption: int,
                 elapsed_time: float):
        self.initial_point = initial_point
        self.optimal_point = optimal_point
        self.iterations = iterations
        self.trajectory = trajectory
        self.cause = cause
        self.oracle = oracle
        self.name = name
        self.memory_consumption = memory_consumption
        self.elapsed_time = elapsed_time

    def print_info(self, print_trajectory: bool = False):
        print(f"""
        {self.name} on {self.initial_point} and {self.oracle.representation}:
        Cause: {self.cause}
        Optimal point: {self.optimal_point}
        Function value at point: {self.oracle.f(*self.optimal_point)}
        Iterations passed: {self.iterations} 
        Memory: {self.memory_consumption} B
        Elapsed time: {self.elapsed_time} s
        {f"Trajectory: {self.trajectory}" if print_trajectory else ""}""")
