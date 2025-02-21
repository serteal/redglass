from typing import List, Optional, Tuple

import numpy as np
import torch as t
from loguru import logger
from tqdm.autonotebook import tqdm

import wandb
from redglass.models import HuggingFaceModel

from .base import AttackConfig, AttackResult, BaseAttack


class EmbeddingAttack(BaseAttack):
    def __init__(
        self,
        model: HuggingFaceModel,
        config: Optional[AttackConfig] = None,
    ):
        if config is None:
            config = AttackConfig()
        super().__init__(model, config)

    def _prepare_chat_template(self, input: List[str]) -> List[str]:
        # Convert to standard huggingface chat format
        messages_list = [[{"role": "user", "content": el}] for el in input]

        # Add optimization token to the last message if not already present
        for messages in messages_list:
            if not any(["{optim_str}" in d["content"] for d in messages]):
                messages[-1]["content"] = messages[-1]["content"] + self.model.optim_token
            else:
                messages[-1]["content"] = messages[-1]["content"].replace(
                    "{optim_str}", self.model.optim_token
                )

        template = self.model.tokenizer.apply_chat_template(
            messages_list, tokenize=False, add_generation_prompt=True
        )
        return template

    def _tokenize_inputs(self, input: List[str], target: List[str]) -> t.Tensor:
        tokenizer_kwargs = {
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
            "truncation": True,
            "max_length": 10000,
        }
        input_tokens = self.model.tokenizer(
            input,
            padding_side="left",
            **tokenizer_kwargs,
        )
        target_tokens = self.model.tokenizer(
            target,
            padding_side="right",
            **tokenizer_kwargs,
        )

        return input_tokens, target_tokens

    def _split_inputs_on_optim_token(
        self, input_tokens: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        optim_positions = (input_tokens.input_ids == self.model.optim_token_id).nonzero()

        # Split into before and after based on optim token position
        before_ids = input_tokens.input_ids[:, : optim_positions[0, 1]].to(self.model.device)
        after_ids = input_tokens.input_ids[:, optim_positions[0, 1] + 1 :].to(self.model.device)
        before_mask = input_tokens.attention_mask[:, : optim_positions[0, 1]].to(self.model.device)
        after_mask = input_tokens.attention_mask[:, optim_positions[0, 1] + 1 :].to(
            self.model.device
        )

        return before_ids, after_ids, before_mask, after_mask

    def run(self, dataloader: t.utils.data.DataLoader) -> AttackResult:
        if self.config.use_wandb:
            wandb.init(project=self.config.wandb_project, config=self.config)

        embedding_layer = self.model.model.get_input_embeddings()

        optim_ids = self.model.tokenizer(
            self.config.optim_str_init, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.model.device)
        optim_embeds = embedding_layer(optim_ids).detach().clone().requires_grad_()

        optimizer = t.optim.Adam([optim_embeds], lr=self.config.lr)

        losses = []
        for epoch in range(self.config.epochs):
            for batch in tqdm(dataloader):
                input, target = batch
                batch_size = len(input)

                template = self._prepare_chat_template(input)

                input_tokens, target_tokens = self._tokenize_inputs(template, target)

                before_ids, after_ids, before_mask, after_mask = self._split_inputs_on_optim_token(
                    input_tokens
                )

                target_ids = target_tokens.input_ids.to(self.model.device)
                target_mask = target_tokens.attention_mask.to(self.model.device)

                # Embed everything that doesn't get optimized
                before_embeds = embedding_layer(before_ids)
                after_embeds = embedding_layer(after_ids)
                target_embeds = embedding_layer(target_ids)

                batch_losses = []
                for _ in range(self.config.max_steps):
                    optimizer.zero_grad()

                    input_embeds = t.cat(
                        [
                            before_embeds.detach(),
                            optim_embeds.expand(batch_size, -1, -1),
                            after_embeds.detach(),
                            target_embeds.detach(),
                        ],
                        dim=1,
                    )

                    optim_mask = t.ones(
                        batch_size,
                        optim_embeds.shape[1],
                        device=self.model.device,
                    )
                    input_attn_mask = t.cat(
                        [
                            before_mask,
                            optim_mask,
                            after_mask,
                            target_mask,
                        ],
                        dim=1,
                    )

                    # TODO: Implement KV Cache
                    output = self.model.forward(
                        input_embeds,
                        attention_mask=input_attn_mask,
                        output_hidden_states=True,
                    )
                    logits = output.logits

                    # Shift logits so token n-1 predicts token n
                    shift = input_embeds.shape[1] - target_ids.shape[1]
                    shift_logits = logits[
                        ..., shift - 1 : -1, :
                    ].contiguous()  # (1, num_target_ids, vocab_size)
                    shift_labels = target_ids

                    loss = t.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=self.model.tokenizer.pad_token_id,  # Ignore padding tokens
                    )
                    batch_losses.append(loss.item())

                    loss.backward()
                    optimizer.step()

                if self.config.use_wandb:
                    wandb.log({"embedding_attack/loss": batch_losses[-1]})

                losses.append(batch_losses)

        return AttackResult(
            losses=losses,
            best_losses=[],
            best_outputs=[],
            suffixes=[],
        )
