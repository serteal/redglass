from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    def __init__(self, data: Sequence[Any]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]
