from abc import ABC, abstractmethod
from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class DetectorConfig: ...


class Detector(ABC):
    def __init__(self, config: DetectorConfig):
        pass

    @abstractmethod
    def fit(self, reps: Float[Tensor, "batch_size seq_len"]):
        pass

    @abstractmethod
    def predict(self, reps: Float[Tensor, "batch_size seq_len"]):
        pass
