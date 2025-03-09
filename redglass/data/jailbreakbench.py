from collections import defaultdict
from typing import Any, cast

import datasets

from redglass.data.base import (
    Dialogue,
    Message,
    TargetedDialogueDataset,
    TargetedDialogueDataType,
)
from redglass.types import Label


class JailbreakBenchDataset(TargetedDialogueDataset):
    base_name = "jailbreak_bench"

    def _get_dialogues(
        self,
    ) -> TargetedDialogueDataType:
        """
        Get all dialogues from JailbreakBench along with target completions.
        """
        ds: datasets.Dataset = datasets.load_dataset("walledai/JailbreakBench")["train"]  # type: ignore
        dialogues: list[Dialogue] = []
        target_completions: list[str] = []
        metadata: dict[str, list[Any]] = defaultdict(list)
        labels: list[Label] = []
        for row in ds:
            dialogue = [
                Message("user", cast(str, row["prompt"]), False),  # type: ignore
            ]  # type: ignore
            dialogues.append(dialogue)

            is_unsafe = row["subset"] == "harmful"  # type: ignore
            labels.append(Label.UNSAFE if is_unsafe else Label.SAFE)

            target_completions.append(cast(str, row["target"]))  # type: ignore

            metadata["behavior"].append(cast(str, row["behavior"]))  # type: ignore
            metadata["category"].append(cast(str, row["category"]))  # type: ignore
            metadata["source"].append(cast(str, row["source"]))  # type: ignore

        self.target_completions = target_completions
        return dialogues, labels, metadata, target_completions
