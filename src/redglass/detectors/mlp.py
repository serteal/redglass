from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .base import Detector, DetectorConfig


class MLP(Detector):
    def __init__(self, config: DetectorConfig):
        super().__init__(config)

    def fit(self, reps: Float[Tensor, "batch_size seq_len"]): ...

    def predict(self, reps: Float[Tensor, "batch_size seq_len"]): ...
