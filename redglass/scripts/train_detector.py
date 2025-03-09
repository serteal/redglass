# %%
from typing import Any

from redglass.activations import PairedActivations
from redglass.detectors import get_detector_class
from redglass.models import get_llama3_model_and_tokenizer
from redglass.repository import DatasetRepository

# %%
REPOSITORY = DatasetRepository()

model, tokenizer = get_llama3_model_and_tokenizer(model_name="llama-8b")
train_dataset = REPOSITORY.get("circuit_breakers")
detector_class = get_detector_class(method="lr")

# %%
split_paired_acts = PairedActivations.from_dataset(
    dataset=train_dataset,
    tokenizer=tokenizer,
    model=model,
    detect_only_start_of_turn=False,
    detect_only_last_token=False,
    layers=[3],
)
safe_acts, unsafe_acts = split_paired_acts.get_all_acts()

# %%
detector_kwargs: dict[str, Any] = {"layers": [3]}

detector = detector_class(**detector_kwargs)
detector.fit(unsafe_acts, safe_acts)
