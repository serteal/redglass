from typing import Optional

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseModel, ModelConfig


class HuggingFaceModel(BaseModel):
    def __init__(
        self,
        name_or_path: str,
        config: Optional[ModelConfig] = None,
        device: Optional[str] = None,
    ):
        if config is None:
            config = ModelConfig()
        super().__init__(config)
        if device is None:
            device = (
                "mps"
                if t.backends.mps.is_available()
                else "cuda" if t.cuda.is_available() else "cpu"
            )
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    def forward(self, inputs, **kwargs):
        if isinstance(inputs, t.Tensor) and inputs.dtype == t.float:
            return self.model(inputs_embeds=inputs, **kwargs)
        return self.model(inputs, **kwargs)

    def generate(self, inputs, **kwargs):
        if isinstance(inputs, t.Tensor) and inputs.dtype == t.float:
            return self.model.generate(inputs_embeds=inputs, **kwargs)
        return self.model.generate(inputs, **kwargs)
