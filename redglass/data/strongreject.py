from collections import defaultdict
from typing import Any, cast

import datasets

from redglass.data.base import (
    Dialogue,
    DialogueDataset,
    DialogueDataType,
    Message,
)
from redglass.types import Label


class StrongREJECTDataset(DialogueDataset):
    base_name = "strong_reject"

    def _get_dialogues(
        self,
    ) -> DialogueDataType:
        """
        Get all dialogues from StrongREJECT.
        """
        ds: datasets.Dataset = datasets.load_dataset("walledai/StrongREJECT")["train"]  # type: ignore
        dialogues: list[Dialogue] = []
        metadata: dict[str, list[Any]] = defaultdict(list)

        for row in ds:
            dialogue = [
                Message("user", cast(str, row["prompt"]), False),  # type: ignore
            ]  # type: ignore
            dialogues.append(dialogue)

            metadata["category"].append(cast(str, row["category"]))  # type: ignore
            metadata["source"].append(cast(str, row["source"]))  # type: ignore

        return dialogues, [Label.UNSAFE] * len(dialogues), metadata
