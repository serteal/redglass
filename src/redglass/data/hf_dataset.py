from typing import Tuple

from datasets import load_dataset

from .base import BaseDataset


class HuggingFaceDataset(BaseDataset):
    def __init__(self, dataset_name_or_path: str, split: str, prompt_key: str, target_key: str):
        self.data = load_dataset(dataset_name_or_path, split=split)
        self.prompt_key = prompt_key
        self.target_key = target_key

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.data[idx][self.prompt_key], self.data[idx][self.target_key]


class AdvBenchDataset(HuggingFaceDataset):
    def __init__(self):
        super().__init__("walledai/AdvBench", "train", "prompt", "target")
