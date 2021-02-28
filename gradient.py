from typing import Callable, List, Tuple, Union
from data.oracle import Oracle


def gradient(
        f: Oracle, bounds: List[Tuple[Float, Float]],
        learning_rate: Union[float, Callable[..., float]],
        iterations: int = 1000, eps: float = None,
):
    pass
