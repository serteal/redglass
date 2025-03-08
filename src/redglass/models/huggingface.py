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
        self.name_or_path = name_or_path
        self.device = device

        # Setup model
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=self.config.dtype,
        ).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.tokenizer.padding_side = "left"  # Pad from left as we are adding suffixes
        if self.tokenizer.pad_token:
            pass
        elif self.tokenizer.unk_token:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        elif self.tokenizer.eos_token:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.optim_token = "<|optim-loc|>"
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.optim_token]})
        self.optim_token_id = self.tokenizer.convert_tokens_to_ids(self.optim_token)

    def forward(self, inputs, **kwargs):
        if isinstance(inputs, t.Tensor) and inputs.dtype in [t.float, t.bfloat16]:
            return self.model(inputs_embeds=inputs, **kwargs)
        return self.model(inputs, **kwargs)

    def generate(self, inputs, **kwargs):
        if isinstance(inputs, t.Tensor) and inputs.dtype in [t.float, t.bfloat16]:
            return self.model.generate(inputs_embeds=inputs, **kwargs)
        return self.model.generate(inputs, **kwargs)
