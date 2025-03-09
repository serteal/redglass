from typing import cast

import pandas as pd

from redglass.data.base import (
    Dialogue,
    Message,
    TargetedDialogueDataset,
    TargetedDialogueDataType,
)
from redglass.types import Label
from redglass.utils import REPO_ROOT


class CircuitBreakersDataset(TargetedDialogueDataset):
    base_name = "circuit_breakers"

    def _get_dialogues(self) -> TargetedDialogueDataType:
        safe_raw_data, unsafe_raw_data = self._get_raw_data()
        dialogues, labels, _, completions = self._get_all_dialogues(
            safe_raw_data, unsafe_raw_data
        )
        return dialogues, labels, None, completions

    def _get_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the raw data from the paper
        """
        safe_raw_data_path = (
            REPO_ROOT / "data/circuit_breakers/benign_train_no_spec_tokens.csv"
        )
        unsafe_raw_data_path = (
            REPO_ROOT / "data/circuit_breakers/harmful_train_no_spec_tokens.csv"
        )
        with open(safe_raw_data_path) as f:
            safe_raw_data = pd.read_csv(f)  # type: ignore
        with open(unsafe_raw_data_path) as f:
            unsafe_raw_data = pd.read_csv(f)  # type: ignore
        return safe_raw_data, unsafe_raw_data

    def _get_all_dialogues(
        self, safe_raw_data: pd.DataFrame, unsafe_raw_data: pd.DataFrame
    ) -> TargetedDialogueDataType:
        """
        Get all dialogues from the raw data, alternating between safe and unsafe
        """
        all_dialogues: list[Dialogue] = []
        all_labels: list[Label] = []
        all_completions: list[str] = []

        for _, row in safe_raw_data.iterrows():
            dialogue: Dialogue = [
                Message("user", cast(str, row["prompt"]).strip(), False),
            ]
            all_dialogues.append(dialogue)
            all_labels.append(Label.SAFE)
            all_completions.append(cast(str, row["response"]).strip())

        for _, row in unsafe_raw_data.iterrows():
            dialogue: Dialogue = [
                Message("user", cast(str, row["prompt"]).strip(), False),
            ]
            all_dialogues.append(dialogue)
            all_labels.append(Label.UNSAFE)
            all_completions.append(cast(str, row["response"]).strip())

        return all_dialogues, all_labels, None, all_completions
