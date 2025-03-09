from abc import ABC, abstractmethod
from typing import List, Optional

import torch as t
from nanogcg import GCGConfig
from pydantic import BaseModel


class FLRTConfig(BaseModel):
    p_add: float = 0.5
    p_swap: float = 0.5
    k1: int = 10
    k2: int = 10
    init_len: int = 10
    buffer_size: int = 100


class AttackConfig(BaseModel):
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
    flrt_config: Optional[FLRTConfig] = None


class AttackResult(BaseModel):
    losses: List[List[float]]  # [batch_size, max_step]
    best_losses: List[float]  # [batch_size]
    best_outputs: List[str]  # [batch_size]
    suffixes: List[str] | List[t.Tensor]  # [batch_size]


class BaseAttack(ABC):
    def __init__(
        self,
        model,
        config: AttackConfig,
    ):
        self.model = model
        self.config = config

    @abstractmethod
    def run(self, dataloader: t.utils.data.DataLoader) -> AttackResult:
        pass
