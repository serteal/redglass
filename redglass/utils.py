import functools
import gc
import inspect
from pathlib import Path
from typing import cast

import torch
import yaml
from jaxtyping import Bool, Float
from torch import Tensor

from redglass.types import Dialogue, Message, Role

REPO_ROOT = Path(__file__).parent.parent


# borrowed from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L69
def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


# modified from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L87
def find_executable_batch_size(
    function: callable = None, starting_batch_size: int = 128
):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(
            find_executable_batch_size, starting_batch_size=starting_batch_size
        )

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join(
                [f"{arg}={value}" for arg, value in zip(params[1:], args[1:])]
            )
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    print(f"Decreasing batch size to: {batch_size}")
                else:
                    raise

    return decorator


def process_str_tokens(str_tokens: list[str], escape_newlines: bool = False):
    """
    Turn html-tag like tokens readable tokens.
    """
    replacements = {
        "<bos>": "[bos]",
        "<start_of_turn>": "[start_of_turn]",
        "<end_of_turn>": "[end_of_turn]",
        "<s>": "[s]",
        "</s>": "[/s]",
    }
    if escape_newlines:
        replacements["\n"] = "\\n"

    for k, v in replacements.items():
        str_tokens = [tok.replace(k, v) for tok in str_tokens]

    return str_tokens


def get_dialogue_from_yaml(path: Path) -> Dialogue:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        Message(cast(Role, msg["role"]), cast(str, msg["content"]), False)
        for msg in data["prompt"]
    ]


def preprocess_dialogue(
    dialogue: Dialogue, fold_system: bool = False
) -> list[dict[str, str]]:
    """Concatenate same-role messages, fold system if necessary, and convert to list of dicts.

    Concatenate adjacent same role messages:
        The Dialogue class allows adjacent messages from the same role to allow running the
        detector on only part of the message. We concatenate these together.

    Fold system messages:
        Gemma wasn't trained with a system prompt. Thus we concatenate any system prompt into
        the first user message.

    Convert to list of dicts:
        This is the format expected by the tokenizer's apply_chat_template method. Also
        returns a boolean indicating if the final message is an assistant message.
    """
    processed: list[dict[str, str]] = []
    if fold_system and dialogue and dialogue[0].role == "system":
        processed.append({"role": "user", "content": dialogue[0].content + "\n\n"})
        dialogue = dialogue[1:]

    for message in dialogue:
        if processed and processed[-1]["role"] == message.role:
            processed[-1]["content"] += message.content
        else:
            processed.append({"role": message.role, "content": message.content})

    return processed


def mask_sparse_activations(
    acts: Float[Tensor, "batch seq layer feature"],
    mask: Bool[Tensor, "batch seq"],
) -> Float[Tensor, "toks layer feature"]:
    """
    Sparse tensor indexing to compute acts[mask]
    Args:
        acts: Sparse tensor of shape [batch, seq, layer, feature]
        mask: Boolean tensor of shape [batch, seq]

    Returns:
        Sparse tensor of shape [mask.sum(), layer, feature]
    """
    if not acts.is_sparse:
        raise ValueError("Input tensor must be sparse")

    # Get indices and values
    if not acts.is_coalesced():
        acts = acts.coalesce()
    indices = acts.indices()
    values = acts.values()

    # Create batch-seq index mapping tensor
    # Shape: [batch, seq] containing flattened indices where mask is True
    n_mask_true = cast(int, mask.sum().item())
    flat_indices = torch.full(mask.shape, -1, device=mask.device)
    flat_indices[mask] = torch.arange(n_mask_true, device=mask.device)  # type: ignore

    # Check if batch-seq positions are masked
    keep_mask = flat_indices[indices[0], indices[1]] >= 0

    if not keep_mask.any():
        return torch.sparse_coo_tensor(
            indices=torch.empty(0, 3, device=acts.device),
            values=torch.empty(0, device=acts.device),
            size=(n_mask_true, acts.size(2), acts.size(3)),
            device=acts.device,
        )

    # Get new indices
    new_batch_idx = flat_indices[indices[0, keep_mask], indices[1, keep_mask]]
    new_layer_idx = indices[2, keep_mask]
    new_feature_idx = indices[3, keep_mask]

    # Stack new indices
    new_indices = torch.stack([new_batch_idx, new_layer_idx, new_feature_idx])

    # Create new sparse tensor
    return torch.sparse_coo_tensor(
        indices=new_indices,
        values=values[keep_mask],
        size=(n_mask_true, acts.size(2), acts.size(3)),
        device=acts.device,
    )


def squeeze_sparse_tensor(tensor: Tensor, dim: int) -> Tensor:
    assert tensor.is_sparse
    assert tensor.shape[dim] == 1
    if not tensor.is_coalesced():
        tensor = tensor.coalesce()

    index_index = [i for i in range(len(tensor.shape)) if i != dim]
    new_indexes = tensor.indices()[index_index, :]
    new_size = tuple(tensor.shape[i] for i in index_index)
    return torch.sparse_coo_tensor(
        indices=new_indexes,
        values=tensor.values(),
        size=new_size,
        device=tensor.device,
    )
