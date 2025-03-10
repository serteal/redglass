# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from collections import Counter

import torch

from redglass.activations import PairedActivations
from redglass.data import CircuitBreakersDataset
from redglass.detectors import MMSDetector
from redglass.models import get_llama3_model_and_tokenizer
from redglass.repository import DatasetRepository

# %%
REPOSITORY = DatasetRepository()

model, tokenizer = get_llama3_model_and_tokenizer(model_name="llama-8b")
fit_data = CircuitBreakersDataset(variant="plain", shuffle_upon_init=True)

# %%
n_train = int(len(fit_data) * 0.7)
train_data, test_data = fit_data[:n_train], fit_data[n_train:]
print(Counter(train_data.labels))
print(Counter(test_data.labels))

train_split_acts = PairedActivations.from_dataset(
    train_data, tokenizer, model, enforce_equal_detect_lengths=True
)
test_split_acts = PairedActivations.from_dataset(
    test_data, tokenizer, model, enforce_equal_detect_lengths=True
)

safe_train_acts, unsafe_train_acts = train_split_acts.get_all_acts()
safe_test_acts, unsafe_test_acts = test_split_acts.get_all_acts()

# %%
detector = MMSDetector(layers=list(range(4, 12)))
detector.fit(unsafe_train_acts, safe_train_acts)

test_safe_scores = detector.score(safe_test_acts)
test_unsafe_scores = detector.score(unsafe_test_acts)

mean_safe_scores = torch.tensor(
    [score.mean() for score in test_safe_scores.scores if len(score) > 0]
)
mean_unsafe_scores = torch.tensor(
    [score.mean() for score in test_unsafe_scores.scores if len(score) > 0]
)

accuracy = (mean_safe_scores < mean_unsafe_scores).float().mean()
print(f"Accuracy: {accuracy}")

# %%
