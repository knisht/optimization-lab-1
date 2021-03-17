from typing import List, Union

import numpy as np

from common.oracle import Oracle


class OptimizationResult:
    def __init__(
            self, initial_point: np.ndarray, optimal_point: np.ndarray, iterations: int,
            trajectory: List[np.ndarray], cause: str, oracle: Oracle, name: str,
            memory_consumption: int, elapsed_time: float, arithm_ops: Union[int, str]
    ):
        self.initial_point = initial_point
        self.optimal_point = optimal_point
        self.iterations = iterations
        self.trajectory = trajectory
        self.cause = cause
        self.oracle = oracle
        self.name = name
        self.memory_consumption = memory_consumption
        self.elapsed_time = elapsed_time
        self.arithm_ops = arithm_ops

    def print_info(self, print_trajectory: bool = False):
        head = f'{self.name} on {self.initial_point} and {self.oracle.representation}:'
        info = [
            f'cause: {self.cause}',
            f'optimal point: {self.optimal_point}',
            f'function value at point: {self.oracle.f(*self.optimal_point)}',
            f'iterations passed: {self.iterations}',
            f'memory: {self.memory_consumption} B',
            f'elapsed time: {self.elapsed_time} s',
            f'arithmetic operations: {self.arithm_ops}'
        ]
        if print_trajectory:
            info.append(f'trajectory: {self.trajectory}')

        print(head)
        for s in info:
            print('-', s)
