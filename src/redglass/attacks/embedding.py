from typing import Callable, Optional

import torch as t
from loguru import logger
from tqdm.autonotebook import tqdm

from redglass.data import BaseDataset
from redglass.losses import BaseLoss
from redglass.models import HuggingFaceModel

from .base import AttackConfig, BaseAttack


class EmbeddingAttack(BaseAttack):
    def __init__(
        self,
        model: HuggingFaceModel,
        loss_fn: Optional[BaseLoss] = None,
        config: Optional[AttackConfig] = None,
    ):
        if config is None:
            config = AttackConfig()
        super().__init__(model, config, loss_fn)

    def run(self, dataset: BaseDataset):
        results = []
        for input, target in dataset:
            logger.info(f"Input: {input}")
            logger.info(f"Target: {target}")
            messages = [{"role": "user", "content": input}]

            if not any(["{optim_str}" in d["content"] for d in messages]):
                messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

            template = self.model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Remove the BOS token -- this will get added when tokenizing, if necessary
            if self.model.tokenizer.bos_token and template.startswith(
                self.model.tokenizer.bos_token
            ):
                template = template.replace(self.model.tokenizer.bos_token, "")
            before_str, after_str = template.split("{optim_str}")

            # Tokenize everything that doesn't get optimized
            before_ids = self.model.tokenizer([before_str], padding=False)["input_ids"]
            after_ids = self.model.tokenizer([after_str], add_special_tokens=False)["input_ids"]
            target_ids = self.model.tokenizer([target], add_special_tokens=False)["input_ids"]

            before_ids, after_ids, target_ids = [
                t.tensor(ids, device=self.model.device)
                for ids in (before_ids, after_ids, target_ids)
            ]

            # Embed everything that doesn't get optimized
            embedding_layer = self.model.model.get_input_embeddings()
            before_embeds, after_embeds, target_embeds = [
                embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
            ]

            # Compute the KV Cache for tokens that appear before the optimized tokens
            with t.no_grad():
                output = self.model.forward(before_embeds, use_cache=True)
                prefix_cache = output.past_key_values

            optim_ids = self.model.tokenizer(
                self.config.optim_str_init, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(self.model.device)
            optim_embeds = embedding_layer(optim_ids).detach().clone().requires_grad_()

            optimizer = t.optim.Adam([optim_embeds], lr=self.config.lr)

            losses = []
            for i in tqdm(range(self.config.max_steps)):
                optimizer.zero_grad()
                input_embeds = t.cat(
                    [optim_embeds, after_embeds.detach(), target_embeds.detach()], dim=1
                )

                output = self.model.forward(
                    input_embeds,
                    past_key_values=prefix_cache,
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
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                loss_float = loss.item()
                losses.append(loss_float)

                if self.config.verbose:
                    logger.info(f"Iter: {i} Loss: {loss_float}")

                loss.backward(retain_graph=True)
                optimizer.step()

            results.append(
                {
                    "losses": losses,
                    "optim_embeds": optim_embeds.cpu(),
                    "input_embeds": t.cat(
                        [before_embeds, optim_embeds, after_embeds], dim=1
                    ).cpu(),
                }
            )

        return results
