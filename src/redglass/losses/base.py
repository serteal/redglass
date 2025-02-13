from abc import ABC, abstractmethod
from dataclasses import dataclass


class BaseLoss(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
