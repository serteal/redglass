from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from redglass.data import BaseDataset
from redglass.losses import BaseLoss
from redglass.models import BaseModel


@dataclass
class AttackConfig:
    max_steps: int = 100
    lr: float = 1e-3
    early_stop_loss: float = 0.01
    optim_str_init: str = "This is a test"
    verbose: bool = False


class BaseAttack(ABC):
    def __init__(
        self,
        model: BaseModel,
        config: AttackConfig,
        loss_fn: BaseLoss,
    ):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn

    @abstractmethod
    def run(self, dataset: BaseDataset) -> BaseDataset:
        pass
