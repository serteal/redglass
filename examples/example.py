from redglass.attacks import EmbeddingAttack
from redglass.data import HuggingFaceDataset
from redglass.models import HuggingFaceModel

model = HuggingFaceModel(name_or_path="HuggingFaceTB/SmolLM2-135M-Instruct", device="cpu")
attack = EmbeddingAttack(model)
dataset = HuggingFaceDataset(
    dataset_name_or_path="walledai/AdvBench",
    split="train",
    prompt_key="prompt",
    target_key="target",
)
results = attack.run(
    dataset=dataset,
)

print(results)
