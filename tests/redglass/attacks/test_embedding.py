import pytest
import torch
from torch.utils.data import DataLoader

from redglass.attacks import AttackConfig, IndividualEmbeddingAttack
from redglass.models import HuggingFaceModel


@pytest.fixture
def model():
    return HuggingFaceModel("HuggingFaceTB/SmolLM2-135M-Instruct", device="cpu")


@pytest.fixture
def attack_config():
    return AttackConfig(
        epochs=1,
        max_steps=2,
        lr=0.01,
        optim_str_init="This is a test",
        use_wandb=False,
    )


@pytest.fixture
def embedding_attack(model, attack_config):
    return IndividualEmbeddingAttack(model, attack_config)


MODEL_TEMPLATE = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n{message}<|optim-loc|><|im_end|>\n<|im_start|>assistant\n"


def test_single_input_prepare_chat_template(embedding_attack):
    inputs = ["Hello, how are you?"]
    templates = embedding_attack._prepare_chat_template(inputs)

    assert isinstance(templates, list)
    assert len(templates) == 1
    assert templates[0] == MODEL_TEMPLATE.format(message=inputs[0])


def test_multiple_input_prepare_chat_template(embedding_attack):
    inputs = ["Hello, how are you?", "What is 2+2?"]
    templates = embedding_attack._prepare_chat_template(inputs)

    expected_result = [MODEL_TEMPLATE.format(message=inp) for inp in inputs]
    assert isinstance(templates, list)
    assert len(templates) == 2
    assert templates == expected_result


def test_tokenize_inputs(embedding_attack):
    inputs = ["What is 2+2?"]
    targets = ["The answer is 4"]

    input_tokens, target_tokens = embedding_attack._tokenize_inputs(inputs, targets)

    assert isinstance(input_tokens.input_ids, torch.Tensor)
    assert isinstance(target_tokens.input_ids, torch.Tensor)
    assert input_tokens.input_ids.ndim == 2
    assert target_tokens.input_ids.ndim == 2


def test_split_inputs_on_optim_token(embedding_attack):
    input_text = [f"Hello{embedding_attack.model.optim_token}world"]
    input_tokens = embedding_attack.model.tokenizer(
        input_text,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    before_ids, after_ids, before_mask, after_mask = embedding_attack._split_inputs_on_optim_token(
        input_tokens
    )

    assert isinstance(before_ids, torch.Tensor)
    assert isinstance(after_ids, torch.Tensor)
    assert isinstance(before_mask, torch.Tensor)
    assert isinstance(after_mask, torch.Tensor)


@pytest.mark.slow
def test_run_attack(embedding_attack):
    inputs = ["What is 2+2?", "What is 3+3?"]
    targets = ["4", "6"]

    dataset = list(zip(inputs, targets))
    dataloader = DataLoader(dataset, batch_size=2)

    result = embedding_attack.run(dataloader)

    assert hasattr(result, "losses")
    assert len(result.losses) > 0
    assert isinstance(result.losses[0], list)
