from typing import cast

import datasets

from redglass.data.base import (
    Dialogue,
    DialogueDataset,
    DialogueDataType,
    Message,
)
from redglass.types import Label


class MaliciousInstructDataset(DialogueDataset):
    base_name = "malicious_instruct"

    def _get_dialogues(
        self,
    ) -> DialogueDataType:
        """
        Get all dialogues from MaliciousInstruct along with target completions.
        """
        ds: datasets.Dataset = datasets.load_dataset("walledai/MaliciousInstruct")[
            "train"
        ]  # type: ignore
        dialogues: list[Dialogue] = []
        for row in ds:
            dialogue = [
                Message("user", cast(str, row["prompt"]), False),  # type: ignore
            ]  # type: ignore
            dialogues.append(dialogue)

        return dialogues, [Label.UNSAFE] * len(dialogues), None
