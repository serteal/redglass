from typing import Optional

import torch as t
from loguru import logger
from tqdm.autonotebook import tqdm

from redglass.models.huggingface import HuggingFaceModel

from .base import AttackConfig, AttackResult, BaseAttack


class FLRTAttack(BaseAttack):
    def __init__(self, model: HuggingFaceModel, config: Optional[AttackConfig] = None):
        if config is None:
            config = AttackConfig()
        super().__init__(model, config)

    def run(self, dataloader: t.utils.data.DataLoader) -> AttackResult:
        pass
