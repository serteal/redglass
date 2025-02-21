from typing import Optional

import nanogcg
import torch as t
from loguru import logger
from nanogcg import GCGConfig
from tqdm.autonotebook import tqdm

from redglass.models.huggingface import HuggingFaceModel

from .base import AttackConfig, AttackResult, BaseAttack


class GCGAttack(BaseAttack):
    def __init__(
        self,
        model: HuggingFaceModel,
        config: Optional[AttackConfig] = None,
    ):
        if config is None:
            config = AttackConfig()
        if config.gcg_config is None:
            config.gcg_config = nanogcg.GCGConfig()
        super().__init__(model, config)

    def run(self, dataloader: t.utils.data.DataLoader) -> AttackResult:
        results = []
        assert dataloader.batch_size == 1, "GCGAttack only supports batch size 1"
        for batch in tqdm(dataloader):
            input, target = batch
            result = nanogcg.run(
                self.model.model,
                self.model.tokenizer,  # type: ignore
                input[0],
                target[0],
                self.config.gcg_config,
            )
            results.append(result)
        return AttackResult(
            losses=[],
            best_losses=[],
            best_outputs=[],
            suffixes=[],
        )
