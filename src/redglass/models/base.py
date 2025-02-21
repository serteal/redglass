from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch as t


@dataclass
class ModelConfig:
    dtype: t.dtype = t.bfloat16


class BaseModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def forward(self, inputs, **kwargs):
        raise NotImplementedError(
            f"Forward pass not implemented or not supported for {self.__class__.__name__}"
        )

    @abstractmethod
    def generate(self, inputs, **kwargs):
        raise NotImplementedError(
            f"Generate method not implemented or not supported for {self.__class__.__name__}"
        )
