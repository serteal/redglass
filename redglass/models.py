from enum import Enum
from functools import cache
from pathlib import Path
from typing import Literal, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from redglass.log import logger

# We only use the instruct variations of the models. Note not all sizes are available for all model
# families.
ModelType = Literal["gemma", "llama"]
ModelSize = Literal[2, 7, 8, 9, 70]


class ModelName(str, Enum):
    GEMMA_2B = "gemma-2b"
    GEMMA_7B = "gemma-7b"
    GEMMA_9B = "gemma-9b"
    LLAMA_8B = "llama-8b"
    LLAMA_70B = "llama-70b"
    LLAMA_70B_3_3 = "llama-70b-3.3"

    @property
    def type(self) -> ModelType:
        return cast(ModelType, self.value.split("-")[0])

    @property
    def size(self) -> ModelSize:
        size_str = self.value.split("-")[1].replace("b", "")
        return cast(ModelSize, int(size_str))

    @property
    def is_llama_3_3(self) -> bool:
        return self == ModelName.LLAMA_70B_3_3


def get_model_name(model_name: ModelName) -> str:
    return model_name.value


def get_num_layers_for_model(model_name: ModelName) -> int:
    if model_name.type == "gemma":
        n_layers = {2: 26, 7: 32, 9: 42}
    elif model_name.type == "llama":
        n_layers = {8: 32, 70: 80}
    else:
        raise NotImplementedError(f"No number of layers for model {model_name}")
    return n_layers[model_name.size]


def get_model_and_tokenizer(
    model_name: ModelName,
    omit_model: bool = False,
    cut_at_layer: int | None = None,
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]:
    """
    Load the model and tokenizer based on the given parameters.
    """
    if model_name.type == "gemma":
        model, tokenizer = get_gemma_model_and_tokenizer(model_name, omit_model)
    elif model_name.type == "llama":
        model, tokenizer = get_llama3_model_and_tokenizer(model_name, omit_model)
    else:
        raise ValueError(f"Invalid model type: {model_name.type}")

    if cut_at_layer is not None and model is not None:
        model.model.layers = model.model.layers[:cut_at_layer]

    return model, tokenizer


@cache
def get_gemma_model_and_tokenizer(
    model_name: ModelName,
    omit_model: bool = False,
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]:
    """
    Loads and returns the model and tokenizer based on the given parameters.

    Args:
        model_name: The Gemma model to load (must be a GEMMA_* variant)
        omit_model: Don't load the model if we are just doing analysis
        pw_locked: Whether to use the pw_locked version
        lora_path: Optional path to a LoRA adapter
    """
    if not model_name.type == "gemma":
        raise ValueError(f"Expected Gemma model, got {model_name}")

    # Determine the appropriate dtype based on model size
    dtype = torch.float16 if model_name.size < 9 else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it", padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.bos_token_id
    if omit_model:
        return None, tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        f"google/gemma-2-{model_name.size}b-it",
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=dtype,
        cache_dir=Path("/data/huggingface"),
        local_files_only=True,
    )

    logger.info(f"Loaded model {model_name}")

    if model.device == torch.device("cpu"):
        logger.warning("Model is on CPU")

    model.eval()
    return model, tokenizer


@cache
def get_llama3_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.bos_token_id
    return tokenizer


@cache
def get_llama3_model_and_tokenizer(
    model_name: ModelName,
    omit_model: bool = False,
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]:
    """Return Llama3 model and tokenizer"""
    tokenizer = get_llama3_tokenizer()
    if omit_model:
        return None, tokenizer

    model_path = {
        ModelName.LLAMA_8B: "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ModelName.LLAMA_70B: "meta-llama/Meta-Llama-3.1-70B-Instruct",
        ModelName.LLAMA_70B_3_3: "meta-llama/Llama-3.3-70B-Instruct",
    }[model_name]
    models_directory = Path("/data/huggingface")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=models_directory,
        local_files_only=True,
    )
    model.eval()

    return model, tokenizer
