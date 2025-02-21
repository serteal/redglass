from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch as t
import transformers
from nanogcg import GCGConfig

from redglass.losses import BaseLoss
from redglass.models import HuggingFaceModel


@dataclass
class AttackConfig:
    epochs: int = 2
    max_steps: int = 10
    lr: float = 1e-3
    early_stop_loss: float = 0.01
    optim_str_init: str = "This is a test"
    verbose: bool = True
    use_wandb: bool = False
    wandb_project: str = "redglass"
    seed: Optional[int] = None
    use_prefix_cache: bool = False
    gcg_config: Optional[GCGConfig] = None


@dataclass
class AttackResult:
    losses: List[List[float]]  # [batch_size, max_step]
    best_losses: List[float]  # [batch_size]
    best_outputs: List[str]  # [batch_size]
    suffixes: List[str] | List[t.Tensor]  # [batch_size]


class BaseAttack(ABC):
    def __init__(
        self,
        model: HuggingFaceModel,
        config: AttackConfig,
    ):
        self.model = model
        self.config = config

    @abstractmethod
    def run(self, dataloader: t.utils.data.DataLoader) -> AttackResult:
        pass
