"""
Train an embedding attack to append optimized tokens to prompts to make
a model generate specific target completions.
"""

# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from transformers import set_seed

from redglass.attacks.embedding import EmbeddingAttack
from redglass.data import CircuitBreakersDataset
from redglass.log import logger
from redglass.models import get_llama3_model_and_tokenizer

set_seed(42)

# Load model and tokenizer
model, tokenizer = get_llama3_model_and_tokenizer(model_name="llama-8b")

# %%
attack = EmbeddingAttack(
    model=model,  # type: ignore
    tokenizer=tokenizer,
    learning_rate=1e-3,
    num_iterations_per_example=10,
    num_epochs=1,
)

# Get training dataset
train_dataset = CircuitBreakersDataset(variant="plain", shuffle_upon_init=True)
train_dataset = train_dataset.get_unsafe()  # we only train on unsafe examples
logger.info(f"Training attack on {len(train_dataset)} examples")

attack.optimize(train_dataset)

# %%
