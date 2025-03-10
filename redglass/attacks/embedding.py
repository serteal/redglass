"""
Embedding-based attacks that optimize tokens to be appended to prompts.
"""

from typing import Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from redglass.data import DialogueDataset, TargetedDialogueDataset
from redglass.data.tokenized_data import TokenizedDataset
from redglass.log import logger
from redglass.types import Dialogue


class EmbeddingAttack:
    """
    An attack that optimizes a set of embeddings that are appended to prompts to make
    the model generate specific target completions.

    Attributes:
        model: The model to attack
        tokenizer: The tokenizer for the model
        num_tokens: Number of optimization tokens to append
        learning_rate: Learning rate for optimization
        num_iterations_per_example: Number of optimization iterations per example
        device: Device to run optimization on
        embedding_matrix: The optimized embeddings (filled after optimize() is called)
        token_ids: The token IDs corresponding to the optimized embeddings
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        optimization_init: str = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
        learning_rate: float = 0.1,
        num_iterations_per_example: int = 10,
        num_epochs: int = 1,
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding attack.

        Args:
            model: The model to attack
            tokenizer: The tokenizer for the model
            num_tokens: Number of optimization tokens to append
            learning_rate: Learning rate for optimization
            num_iterations_per_example: Number of optimization iterations per example
            device: Device to run optimization on (defaults to CUDA if available)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimization_init = optimization_init
        self.learning_rate = learning_rate
        self.num_iterations_per_example = num_iterations_per_example
        self.num_epochs = num_epochs
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = self.model.to(self.device)

    def _prepare_dataset(
        self, dataset: TargetedDialogueDataset
    ) -> Tuple[TokenizedDataset, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare the dataset for optimization by creating prompt dialogues and tokenizing.

        Args:
            dataset: The dataset to prepare

        Returns:
            Tuple of (tokenized_prompts, tokenized_targets)
        """
        prompt_dialogues = []
        for i, dialogue in enumerate(dataset.dialogues):
            # Create a copy of the dialogue without the last assistant message
            prompt_dialogue = []
            for msg in dialogue:
                if msg.role == "assistant" and msg is dialogue[-1]:
                    # Skip the last assistant message as this is what we want to generate
                    continue
                prompt_dialogue.append(msg)

            prompt_dialogues.append(Dialogue(prompt_dialogue))

        # Tokenize prompts
        tokenized_prompts = TokenizedDataset.from_dialogue_list(
            prompt_dialogues,
            self.tokenizer,
            detection_mask_padding=dataset.padding,
            for_generation=True,  # Set to True to prepare for generation
        )

        # Tokenize targets
        tokenized_targets = self.tokenizer(
            dataset.target_completions,
            return_tensors="pt",
            padding=True,
            padding_side="right",  # since we're adding to the end of the sequence
            add_special_tokens=False,  # not add <begin_of_text>
        ).to(self.device)

        return tokenized_prompts, tokenized_targets

    def optimize(self, dataset: DialogueDataset) -> dict[int, torch.Tensor]:
        """
        Optimize embedding tokens for the given dataset.
        Creates individual optimized embeddings for each example.

        Args:
            dataset: Dataset containing prompt-target pairs

        Returns:
            Dictionary mapping example indices to their optimized embeddings
        """
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        embedding_layer = self.model.get_input_embeddings()
        tokenized_prompts, tokenized_targets = self._prepare_dataset(dataset)

        optimized_tokens = self.tokenizer(
            self.optimization_init,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)
        optimized_embeddings = (
            embedding_layer(optimized_tokens).detach().clone().requires_grad_()
        )
        optimizer = torch.optim.Adam(
            [optimized_embeddings], lr=self.learning_rate, eps=1e-6
        )

        for epoch in range(self.num_epochs):
            for idx in tqdm(range(len(tokenized_prompts)), desc="Example"):
                # Tokenize inputs
                input_tokens = (
                    tokenized_prompts.tokens[idx].to(self.device).unsqueeze(0)
                )
                input_attention_mask = (
                    tokenized_prompts.attention_mask[idx].to(self.device).unsqueeze(0)
                )
                input_embeddings = embedding_layer(input_tokens)

                # Tokenize target
                target = tokenized_targets.input_ids[idx].to(self.device).unsqueeze(0)
                target_attention_mask = (
                    tokenized_targets.attention_mask[idx].to(self.device).unsqueeze(0)
                )
                target_embeddings = embedding_layer(target)

                losses = []
                for _ in tqdm(range(self.num_iterations_per_example)):
                    optimizer.zero_grad()

                    combined_embeddings = torch.cat(
                        [
                            input_embeddings.detach(),
                            optimized_embeddings,
                            target_embeddings.detach(),
                        ],
                        dim=1,
                    )
                    combined_attention_mask = torch.cat(
                        [
                            input_attention_mask.detach(),
                            torch.ones(
                                (1, optimized_tokens.shape[1]), device=self.device
                            ),
                            target_attention_mask.detach(),
                        ],
                        dim=1,
                    )

                    outputs = self.model(
                        inputs_embeds=combined_embeddings,
                        attention_mask=combined_attention_mask,
                        output_hidden_states=True,
                    )
                    logits = outputs.logits

                    # Shift logits so token n-1 predicts token n
                    shift = combined_embeddings.shape[1] - target.shape[1]
                    shift_logits = logits[..., shift - 1 : -1, :].contiguous()
                    shift_labels = target

                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    loss_float = loss.item()
                    losses.append(loss_float)

                    loss.backward()
                    optimizer.step()

                logger.info(f"Epoch {epoch} Loss: {sum(losses) / len(losses)}")

        return None

    # def optimize(self, dataset: TargetedDialogueDataset) -> list[float]:
    #     """
    #     Optimize embedding tokens for the given dataset.
    #     Creates individual optimized embeddings for each example.

    #     Args:
    #         dataset: Dataset containing prompt-target pairs

    #     Returns:
    #         Dictionary mapping example indices to their optimized embeddings
    #     """
    #     if len(dataset) == 0:
    #         raise ValueError("Dataset is empty")

    #     embedding_layer = self.model.get_input_embeddings()

    #     losses = []
    #     for i, (dialogue, target) in enumerate(
    #         zip(dataset.dialogues, dataset.target_completions)
    #     ):
    #         prompt_dialogue = []
    #         for msg in dialogue:
    #             if msg.role == "assistant" and msg is dialogue[-1]:
    #                 # Skip the last assistant message as this is what we want to generate
    #                 continue
    #             prompt_dialogue.append(msg)

    #         messages = preprocess_dialogue(prompt_dialogue)
    #         print("=" * 100)
    #         print(messages)
    #         print(target)
    #         print("=" * 100)

    #         if not any(["{optim_str}" in d["content"] for d in messages]):
    #             messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

    #         template = self.tokenizer.apply_chat_template(
    #             messages, tokenize=False, add_generation_prompt=True
    #         )
    #         # Remove the BOS token -- this will get added when tokenizing, if necessary
    #         if self.tokenizer.bos_token and template.startswith(
    #             self.tokenizer.bos_token
    #         ):
    #             template = template.replace(self.tokenizer.bos_token, "")
    #         before_str, after_str = template.split("{optim_str}")

    #         # Tokenize everything that doesn't get optimized
    #         before_ids = self.tokenizer([before_str], padding=False)["input_ids"]
    #         after_ids = self.tokenizer([after_str], add_special_tokens=False)[
    #             "input_ids"
    #         ]
    #         target_ids = self.tokenizer([target], add_special_tokens=False)["input_ids"]

    #         before_ids, after_ids, target_ids = [
    #             torch.tensor(ids, device=self.device)
    #             for ids in (before_ids, after_ids, target_ids)
    #         ]

    #         # Embed everything that doesn't get optimized
    #         before_embeds, after_embeds, target_embeds = [
    #             embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
    #         ]

    #         # Compute the KV Cache for tokens that appear before the optimized tokens
    #         with torch.no_grad():
    #             output = self.model(inputs_embeds=before_embeds, use_cache=True)
    #             prefix_cache = output.past_key_values

    #         optim_ids = self.tokenizer(
    #             self.optimization_init, return_tensors="pt", add_special_tokens=False
    #         )["input_ids"].to(self.device)
    #         optim_embeds = embedding_layer(optim_ids).detach().clone().requires_grad_()

    #         optimizer = torch.optim.Adam(
    #             [optim_embeds], lr=self.learning_rate, eps=1e-6
    #         )

    #         losses = []
    #         for i in range(self.num_iterations_per_example):
    #             optimizer.zero_grad()
    #             input_embeds = torch.cat(
    #                 [optim_embeds, after_embeds.detach(), target_embeds.detach()], dim=1
    #             )

    #             output = self.model(
    #                 inputs_embeds=input_embeds,
    #                 past_key_values=prefix_cache,
    #                 output_hidden_states=True,
    #             )
    #             logits = output.logits

    #             # Shift logits so token n-1 predicts token n
    #             shift = input_embeds.shape[1] - target_ids.shape[1]
    #             shift_logits = logits[
    #                 ..., shift - 1 : -1, :
    #             ].contiguous()  # (1, num_target_ids, vocab_size)
    #             shift_labels = target_ids

    #             loss = torch.nn.functional.cross_entropy(
    #                 shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    #             )
    #             loss_float = loss.item()
    #             losses.append(loss_float)

    #             logger.info(f"Loss: {loss_float}")

    #             loss.backward()
    #             optimizer.step()

    #     return losses
