from typing import cast

import datasets

from redglass.data.base import (
    Dialogue,
    DialogueDataset,
    DialogueDataType,
    Message,
)
from redglass.types import Label, Role

RawDialogue = list[dict[str, str]]


class AlpacaDataset(DialogueDataset):
    base_name = "alpaca"

    def _get_dialogues(self) -> DialogueDataType:
        ds: datasets.Dataset = datasets.load_dataset("tatsu-lab/alpaca")["train"]  # type: ignore
        dialogues: list[Dialogue] = []
        ds = ds.take(10_000)
        for row in ds:
            dialogue = [
                Message("system", "You are an AI assistant.", False),
                Message("user", cast(str, row["instruction"]), False),  # type: ignore
            ]  # type: ignore
            if row["input"] != "":  # type: ignore
                dialogue.append(Message("user", cast(str, " " + row["input"]), False))  # type: ignore
            if row["output"] == "":  # type: ignore
                continue
            dialogue.append(Message("assistant", cast(str, row["output"]), True))  # type: ignore
            dialogues.append(dialogue)

        return dialogues, [Label.SAFE] * len(dialogues), None


class LmSysChatDataset(DialogueDataset):
    """Real user-dialogues from https://huggingface.co/datasets/lmsys/lmsys-chat-1m
    Dialogues are with a range of models, although ~half of the dataset is vicuna (a llama finetune)
    """

    base_name = "lmsys"

    def _get_dialogues(self) -> DialogueDataType:
        ds: datasets.Dataset = datasets.load_dataset("lmsys/lmsys-chat-1m")["train"]  # type: ignore
        ds = ds.take(1000)
        dialogues: list[Dialogue] = []
        models: list[str] = ds["model"]
        for conv in cast(list[RawDialogue], ds["conversation"]):
            dialogue: list[Message] = []
            for msg in conv:
                processed_msg = self._process_message(msg)
                if processed_msg is not None:
                    dialogue.append(processed_msg)
            if dialogue:
                dialogues.append(dialogue)
        return dialogues, [Label.SAFE] * len(dialogues), {"models": models}

    @staticmethod
    def _process_message(message: dict[str, str]) -> Message | None:
        if (
            message["role"] not in ["user", "assistant"]
            or message["content"].strip() == ""
        ):
            return None
        return Message(
            role=cast(Role, message["role"]),
            content=message["content"].strip(),
            detect=message["role"] == "assistant",
        )
