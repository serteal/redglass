from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import pandas as pd
from loguru import logger
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    def __init__(self, data: Sequence[Any]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]


class LocalDataset(BaseDataset):
    def __init__(self, data: pd.DataFrame, prompt_key: str, target_key: str):
        self.data = data
        logger.info(f"Loaded {len(self.data)} rows")
        self.prompt_key = prompt_key
        self.target_key = target_key

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        row = self.data.iloc[idx]
        return row[self.prompt_key], row[self.target_key]
