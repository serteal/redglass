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


class XSTestDataset(DialogueDataset):
    base_name = "xstest"

    def _get_dialogues(
        self,
    ) -> DialogueDataType:
        """
        Get all dialogues from AdvBench along with target completions.
        """
        ds: datasets.Dataset = datasets.load_dataset("walledai/XSTest")["test"]  # type: ignore
        dialogues: list[Dialogue] = []
        metadata: dict[str, list[Any]] = defaultdict(list)
        labels: list[Label] = []

        for row in ds:
            dialogue = [
                Message("user", cast(str, row["prompt"]), False),  # type: ignore
            ]  # type: ignore
            dialogues.append(dialogue)

            is_unsafe = row["label"] == "unsafe"  # type: ignore
            labels.append(Label.UNSAFE if is_unsafe else Label.SAFE)

            metadata["focus"].append(cast(str, row["focus"]))  # type: ignore
            metadata["type"].append(cast(str, row["type"]))  # type: ignore
            metadata["note"].append(cast(str, row["note"]))  # type: ignore

        return dialogues, labels, metadata
