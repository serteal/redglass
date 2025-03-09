from typing import cast

import datasets

from redglass.data.base import (
    Dialogue,
    Message,
    TargetedDialogueDataset,
    TargetedDialogueDataType,
)
from redglass.types import Label


class AdvBenchDataset(TargetedDialogueDataset):
    base_name = "adv_bench"

    def _get_dialogues(
        self,
    ) -> TargetedDialogueDataType:
        """
        Get all dialogues from AdvBench along with target completions.
        """
        ds: datasets.Dataset = datasets.load_dataset("walledai/AdvBench")["train"]  # type: ignore
        dialogues: list[Dialogue] = []
        target_completions: list[str] = []

        for row in ds:
            dialogue = [
                Message("user", cast(str, row["prompt"]), False),  # type: ignore
            ]  # type: ignore
            dialogues.append(dialogue)

            target_completions.append(cast(str, row["target"]))  # type: ignore

        self.target_completions = target_completions
        return dialogues, [Label.UNSAFE] * len(dialogues), None, target_completions
