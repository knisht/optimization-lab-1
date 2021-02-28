from typing import Callable, List, Tuple, Union
from data.oracle import Oracle


def gradient(
        f: Oracle, bounds: List[Tuple[Float, Float]],
        it: int = 1000, eps: float = None, learning_rate: float = 0.01,
        step_optimizer: Union[None, Callable[..., float]] = None
):
    pass
