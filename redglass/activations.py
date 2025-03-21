"""
Creates and handles inputs to white box detectors. Run's the model on tokenized input data
and returns activations at the detection positions.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import trange
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from redglass.data import DialogueDataset, TargetedDialogueDataset
from redglass.data.tokenized_data import (
    TokenizedDataset,
    get_tokenized_dataset_pair,
)
from redglass.utils import mask_sparse_activations


@dataclass
class Activations:
    """
    Responsible for running the model and grabbing activations on the "detect" tokens.
    """

    all_acts: Float[Tensor, "batch seqpos layer emb"]
    tokenized_dataset: TokenizedDataset
    logits: None | Float[Tensor, "batch seqpos vocab"] = None
    layers: list[int] | None = None

    @classmethod
    @torch.no_grad()
    def from_model(
        cls,
        model: PreTrainedModel,
        tokenized_dataset: TokenizedDataset,
        save_logits: bool = False,
        batch_size: int = 20,
        layers: list[int] | None = None,
        verbose: bool = False,
    ) -> "Activations":
        """Creates Activations by running the model on the tokenized dataset."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        activations_by_batch: list[Float[Tensor, "batch seqpos layer emb"]] = []
        logits_by_batch: list[Float[Tensor, "batch seqpos vocab"]] = []

        for i in trange(
            0,
            len(tokenized_dataset),
            batch_size,
            desc="Computing activations",
            disable=not verbose,
        ):
            batch: TokenizedDataset = tokenized_dataset[i : i + batch_size]

            all_activations: list[Float[Tensor, "batch seqpos emb"]] = []
            with torch.autocast(
                device_type="cuda", enabled="llama" in model.name_or_path.lower()
            ):
                fwd_out = model(
                    batch.tokens.to(device),
                    output_hidden_states=True,
                    attention_mask=batch.attention_mask.to(device),
                    # only compute logits on last token if we don't care about logits
                    num_logits_to_keep=0 if save_logits else 1,
                    use_cache=False,
                )

                layers_to_process = (
                    layers if layers is not None else range(len(fwd_out.hidden_states))
                )
                for layer_idx in layers_to_process:
                    layer_activation = fwd_out.hidden_states[layer_idx].cpu()
                    all_activations.append(layer_activation)

                if save_logits:
                    logits_by_batch.append(fwd_out.logits.cpu())

                del fwd_out

            activations_by_batch.append(torch.stack(all_activations, dim=2))

        all_acts = torch.cat(activations_by_batch)
        logits = torch.cat(logits_by_batch) if save_logits else None

        return cls(all_acts, tokenized_dataset, logits, layers)

    def get_masked_activations(self) -> Float[Tensor, "toks layer emb"]:
        """Get activations at the detection positions.

        This flattens the batch and seq dimensions into a single 'toks' dimension."""
        assert self.tokenized_dataset.detection_mask is not None
        if self.all_acts.is_sparse:
            return mask_sparse_activations(
                self.all_acts, self.tokenized_dataset.detection_mask
            )
        else:
            return self.all_acts[self.tokenized_dataset.detection_mask]

    def get_all_activations(self) -> Float[Tensor, "toks layer emb"]:
        """Get all activations.

        This flattens the batch and seq dimensions into a single 'toks' dimension."""
        assert self.all_acts is not None
        # converting to bool is important for indexing reasons
        if self.all_acts.is_sparse:
            return mask_sparse_activations(
                self.all_acts, self.tokenized_dataset.attention_mask.bool()
            )
        else:
            return self.all_acts[self.tokenized_dataset.attention_mask.bool()]

    def get_masked_logits(self) -> Float[Tensor, "toks vocab"]:
        """Get logits at the detection positions.

        This flattens the batch and seq dimensions into a single 'toks' dimension."""
        assert self.logits is not None
        return self.logits[self.tokenized_dataset.detection_mask]

    def save(
        self, file_path: str | Path, only_at_mask: bool = False, sparse: bool = False
    ) -> None:
        """Save the Activations object to a file.

        If only_at_mask is True, only the activations at the detection positions are saved. The
        activations at the other positions will be loaded as NaNs.
        """
        logits_to_save = (
            self.get_masked_logits()
            if (only_at_mask and self.logits is not None)
            else self.logits
        )

        if self.all_acts.is_sparse:
            if only_at_mask:
                raise NotImplementedError(
                    "Sparse activations not supported for only_at_mask"
                )

            coalesced_acts = self.all_acts.coalesce()
            acts_to_save = {
                "indices": coalesced_acts.indices().T,
                "values": coalesced_acts.values(),
                "shape": coalesced_acts.shape,
                "sparse": True,
            }
        else:
            acts_to_save = (
                self.get_masked_activations() if only_at_mask else self.all_acts
            )

        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "acts": acts_to_save,
                    "logits": logits_to_save,
                    "tokenized_dataset": self.tokenized_dataset,
                    "only_at_mask": only_at_mask,
                    "layers": self.layers,
                },
                f,
            )

    @classmethod
    def load(cls, file_path: str | Path) -> "Activations":
        """Load an Activations object from a file."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        activations = cls.__new__(cls)
        activations.tokenized_dataset = data["tokenized_dataset"]
        activations.layers = data["layers"]

        if data["only_at_mask"]:
            mask = activations.tokenized_dataset.detection_mask
            assert mask
            if isinstance(data["acts"], dict) and data["acts"].get("sparse"):
                raise NotImplementedError(
                    "Sparse activations not supported for only_at_mask"
                )

            else:
                act_shape = (
                    mask.shape + data["acts"].shape[-2:]
                )  # [batch, seqpos] + [layer, emb]
                activations.all_acts = torch.full(
                    act_shape, float("nan"), dtype=data["acts"].dtype
                )
                activations.all_acts[mask] = data["acts"]

            if data["logits"] is not None:
                logits_shape = mask.shape + (data["logits"].shape[-1],)
                activations.logits = torch.full(logits_shape, float("nan"))
                activations.logits[mask] = data["logits"]
            else:
                activations.logits = None
        else:
            if isinstance(data["acts"], dict) and data["acts"].get("sparse"):
                # Handle sparse format
                activations.all_acts = torch.sparse_coo_tensor(
                    indices=data["acts"]["indices"]
                    .long()
                    .t(),  # Convert to long and transpose
                    values=data["acts"]["values"],
                    size=data["acts"]["shape"],
                )
            else:
                activations.all_acts = data["acts"]
            activations.logits = data["logits"]

        return activations

    def __getitem__(self, idx: int | slice) -> "Activations":
        """
        Implement slicing functionality for the Activations class.
        This allows users to slice the dataset along the batch dimension.
        """
        sliced_activations = Activations.__new__(Activations)
        sliced_activations.all_acts = self.all_acts[idx]
        sliced_activations.logits = (
            self.logits[idx] if self.logits is not None else None
        )
        sliced_activations.tokenized_dataset = self.tokenized_dataset[idx]
        sliced_activations.layers = self.layers
        return sliced_activations

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.tokenized_dataset)

    def to(
        self, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ):
        self.all_acts = self.all_acts.to(dtype=dtype, device=device)
        if self.logits is not None:
            self.logits = self.logits.to(dtype=dtype, device=device)

    def to_dense(self):
        """Converts sparse activations to dense in place."""
        if self.all_acts.is_sparse:
            self.all_acts = self.all_acts.to_dense()

    def apply_sae_latent_mask(self, whitelist: list[int]):
        """Zero out all activations except for the whitelisted latents."""
        # activations are a sparse tensor with shape [batch, seq, layer, feature]
        # mask out all but the whitelisted latents
        mask = torch.zeros(self.all_acts.shape[-1], dtype=self.all_acts.dtype)
        mask[whitelist] = 1.0
        self.all_acts = self.all_acts * mask


@dataclass
class PairedActivations:
    """
    Holds safe and unsafe activations for a DialogueDataset or TargetedDialogueDataset.

    This class is intended for training where it is useful to enforce that the safe and
    unsafe dialogues are of the same length as a variance reduction technique.
    """

    dataset: DialogueDataset | TargetedDialogueDataset
    safe_acts: Activations
    unsafe_acts: Activations

    def get_all_acts(self):
        return self.safe_acts, self.unsafe_acts

    @classmethod
    def from_dataset(
        cls,
        dataset: DialogueDataset | TargetedDialogueDataset,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel | None,
        batch_size: int = 12,
        detect_only_start_of_turn: bool = False,
        detect_only_last_token: bool = False,
        layers: list[int] | None = None,
        save_logits: bool = False,
        verbose: bool = False,
        enforce_equal_detect_lengths: bool = False,
    ) -> "PairedActivations":
        """
        Create SplitPairedActivations from a DialogueDataset.

        Args:
            dataset (DialogueDataset): Dataset to process
            tokenizer (PreTrainedTokenizerBase): Tokenizer to use
            model (PreTrainedModel): Model to get activations from
            batch_size (int): Batch size for processing
            detect_only_start_of_turn (bool): Whether to only detect at start of turn tokens
            detect_only_last_token (bool): Whether to only detect at last token before EOT
            layers (list[int] | None): Which layers to get activations from
            save_logits (bool): Whether to save logits
            verbose (bool): Whether to show progress bars
            enforce_equal_detect_lengths (bool): Whether to enforce equal detection lengths
            use_goodfire_sae_acts (bool): Whether to use Goodfire SAE activations
        """
        safe = dataset.get_safe()
        unsafe = dataset.get_unsafe()

        # make sure they are the same length by splicing to the min length
        min_length = min(len(safe), len(unsafe))
        safe = safe[:min_length]
        unsafe = unsafe[:min_length]

        safe_toks, unsafe_toks = get_tokenized_dataset_pair(
            safe,
            unsafe,
            tokenizer,
            enforce_equal_detect_lengths=enforce_equal_detect_lengths,
            detect_only_start_of_turn=detect_only_start_of_turn,
            detect_only_last_token=detect_only_last_token,
        )

        assert model is not None, "Model must be loaded to get activations."
        safe_acts = Activations.from_model(
            model,
            safe_toks,
            layers=layers,
            save_logits=save_logits,
            verbose=verbose,
            batch_size=batch_size,
        )
        unsafe_acts = Activations.from_model(
            model,
            unsafe_toks,
            layers=layers,
            save_logits=save_logits,
            verbose=verbose,
            batch_size=batch_size,
        )

        return cls(
            dataset=dataset,
            safe_acts=safe_acts,
            unsafe_acts=unsafe_acts,
        )
