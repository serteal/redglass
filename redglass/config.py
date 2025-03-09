import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import AfterValidator, BaseModel, Field, field_serializer, field_validator
from tqdm import tqdm, trange
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from redglass.activations import Activations, PairedActivations
from redglass.data import DialogueDataset, TargetedDialogueDataset
from redglass.data.tokenized_data import TokenizedDataset
from redglass.detectors import (
    Detector,
    get_detector_class,
)
from redglass.log import logger
from redglass.models import (
    ModelName,
    get_model_and_tokenizer,
    get_model_name,
    get_num_layers_for_model,
)
from redglass.repository import DatasetRepository
from redglass.scores import Scores
from redglass.utils import REPO_ROOT

REPOSITORY = DatasetRepository()

Reduction = Literal["mean", "max", "relu"]

# we wrap in a lambda to avoid avoid passing model info as kwarg to check_dataset_name
DatasetName = Annotated[
    str, AfterValidator(lambda name: REPOSITORY.check_dataset_name(name))
]
ControlDatasetName = Annotated[
    str, AfterValidator(lambda name: REPOSITORY.check_dataset_name(name))
]


class ExperimentConfig(BaseModel):
    """
    Pydantic model for storing the configuration of an experiment.
    Saved to file alongside the experiment, except for the folder and name, which are automatically
    created at runtime.

    Attributes:
        # Core experiment configuration
        method: The method used to train the detector
        model_name: Which family of model to use, e.g. gemma, llama - we always use
            instruction tuned models

        # Dataset configuration
        train_data: The dataset used to train the detector in "base_name__variant" format
        eval_data: The datasets used to evaluate the detector in "base_name__variant" format
        control_data: The control datasets used in "base_name__variant" format
        trim_reasoning: Whether to trim reasoning from the datasets. If True, removes any reasoning
            text from the responses.

        # Training and evaluation settings
        train_on_policy: Whether to train on on policy data
        eval_on_policy: Whether to evaluate on on policy data
        control_on_policy: Whether to use on-policy data for control datasets
        detect_only_start_of_turn: Whether to only detect at the start of turn tokens
        detect_only_last_token: Whether to only detect at the last token before EOT
        val_fraction: The fraction of training data to use for validation
        control_dataset_size: The number of dialogues in the control dataset

        # Other
        detect_layers: The layers to use for the detector. By default, the middle half of the layers
            are used.
        lora_path: The path to a LORA finetune to attach to the model
        reg_coeff: The regularization coefficient for logistic regression methods (lr and mlr)

        # Experiment identification and storage
        id: Optional id prepended to the experiment name
        timestamp: The timestamp of the experiment, automatically generated
        name: The name of the experiment, automatically generated
        folder: The folder where the experiment is saved, automatically generated
    """

    # Core experiment configuration
    method: Literal["mms", "lat", "lr", "mlp", "cmms", "mlr", "best"]
    model_name: ModelName

    # Dataset configuration
    train_data: DatasetName = "repe_honesty"
    eval_data: list[DatasetName] = Field(default_factory=list)
    control_data: list[DatasetName] = Field(default_factory=list)
    trim_reasoning: bool = True

    # Training and evaluation settings
    train_on_policy: bool = False
    eval_on_policy: bool = True
    control_on_policy: bool = False
    detect_only_start_of_turn: bool = False
    detect_only_last_token: bool = False
    val_fraction: float = Field(default=0.2, ge=0, le=1)
    control_dataset_size: int = 1000

    # Detector configuration
    detect_layers: list[int] = Field(
        default_factory=list
    )  # if empty, will be set to middle half
    detect_num_latents: int | None = None

    # Other
    lora_path: str | Path | None = None
    reg_coeff: float | None = None
    normalize_acts: bool = True
    max_llama_token_length: int | None = None

    # Experiment identification and storage
    id: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    name: str = ""
    folder: Path = Field(default_factory=lambda: REPO_ROOT / "results")

    @field_validator("model_name", mode="before")
    @classmethod
    def validate_model_name(cls, v: str | ModelName) -> ModelName:
        if isinstance(v, ModelName):
            return v
        try:
            return ModelName(v)
        except ValueError as e:
            raise ValueError(
                f"Invalid model name: {v} not in {[m.value for m in ModelName]}"
            ) from e

    def model_post_init(self, _context: Any):
        self._set_default_name()
        self._check_train_data_not_in_eval_data()
        self._set_default_detect_layers()
        self._validate_reg_coeff()
        self._validate_detector_config()
        self._setup_folder_and_config()

    @classmethod
    def from_path(cls, folder: Path):
        """Load config file within the folder (either json or yaml)"""
        if (folder / "cfg.yaml").exists():
            with open(folder / "cfg.yaml") as f:
                config_data = yaml.safe_load(f)
        else:
            assert (folder / "cfg.json").exists(), (
                f"No config file found in folder {folder}"
            )
            with open(folder / "cfg.json") as f:
                config_data = json.load(f)

        config_data["folder"] = folder.parent
        config_data["name"] = folder.name

        # Handle old config format
        if "type_model" in config_data:
            # Convert old format to new ModelName
            type_model = config_data.pop("type_model")
            size_model = config_data.pop("size_model")
            use_llama_3_3 = config_data.pop("use_llama_3_3", False)

            # Construct model name string
            model_str = f"{type_model}-{size_model}b"
            if use_llama_3_3 and type_model == "llama" and size_model == 70:
                model_str += "-3.3"

            # Convert to ModelName enum
            config_data["model_name"] = model_str

        return cls(**config_data)

    def _get_default_detect_layers(self) -> list[int]:
        n_layers = get_num_layers_for_model(self.model_name)
        return list(range(int(0.25 * n_layers), int(0.75 * n_layers)))

    def _get_default_name(self):
        name = f"{self.method}__{self.train_data}__{self.model_name.value}__{self.timestamp}"
        if self.id:
            name = f"{self.id}_{name}"
        name = name.replace("/", "_")
        return name

    def _set_default_name(self) -> None:
        if not self.name:
            self.name = self._get_default_name()

    def _check_train_data_not_in_eval_data(self) -> None:
        if self.train_data in self.eval_data:
            raise ValueError(f"Train data {self.train_data} is also in eval data")

    def _set_default_detect_layers(self) -> None:
        if not self.detect_layers:
            self.detect_layers = self._get_default_detect_layers()

    def _validate_reg_coeff(self) -> None:
        """Validate that reg_coeff is only set for lr and mlr methods"""
        if self.reg_coeff is not None and self.method not in ["lr", "mlr"]:
            raise ValueError("reg_coeff can only be set for lr and mlr methods")

    def _setup_folder_and_config(self) -> None:
        self.folder = self.folder / self.name
        self.folder.mkdir(parents=True, exist_ok=True)
        if not (self.folder / "cfg.yaml").exists():
            self._save_config()

    @field_serializer("model_name")
    def serialize_model_name(self, model_name: ModelName) -> str:
        return model_name.value

    def _save_config(self):
        with open(self.folder / "cfg.yaml", "w") as f:
            config_dict = self.model_dump(exclude={"name", "folder"})
            yaml.safe_dump(config_dict, f, indent=2)

    def _validate_detector_config(self) -> None:
        if self.method == "best" and self.detect_num_latents is None:
            raise ValueError("detect_num_latents must be set for best latents detector")


class Experiment:
    cfg: ExperimentConfig
    tokenizer: PreTrainedTokenizerBase
    _model: PreTrainedModel | None

    def __init__(
        self,
        cfg: ExperimentConfig,
    ):
        """Initializes the experiment.

        Args:
            cfg (ExperimentConfig): Configuration for the experiment.
        """
        self.cfg = cfg
        self._model = None  # Store actual model in _model attribute
        _, tokenizer = get_model_and_tokenizer(cfg.model_name, omit_model=True)
        self.tokenizer = tokenizer
        self.train_dataset = self.get_train_dataset()
        self.eval_datasets = self.get_eval_datasets()
        self.control_datasets = self.get_control_datasets()
        assert self.cfg.folder.exists(), f"Folder {self.cfg.folder} does not exist"

    # ----- DETECTOR ----- #

    def get_detector(self) -> Detector:
        """Loads from saved file if possible, or trains a new one and saves if not."""

        path = self.cfg.folder / "detector.pt"
        detector_class = get_detector_class(self.cfg.method)

        if path.exists():
            return detector_class.load(path)

        logger.info("Training detector")

        split_paired_acts = PairedActivations.from_dataset(
            dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            model=self.model,
            detect_only_start_of_turn=self.cfg.detect_only_start_of_turn,
            detect_only_last_token=self.cfg.detect_only_last_token,
            layers=self.cfg.detect_layers,
        )
        safe_fit_acts, unsafe_fit_acts = split_paired_acts.get_all_acts()

        detector_kwargs: dict[str, Any] = {"layers": self.cfg.detect_layers}
        if self.cfg.method == "best":
            detector_kwargs["detect_num_latents"] = self.cfg.detect_num_latents
            detector_kwargs["experiment_path"] = self.cfg.folder
        if self.cfg.reg_coeff is not None:
            detector_kwargs["reg_coeff"] = self.cfg.reg_coeff
        if self.cfg.method == "lr":
            detector_kwargs["normalize"] = self.cfg.normalize_acts
        detector = detector_class(**detector_kwargs)
        detector.fit(unsafe_fit_acts, safe_fit_acts)
        out_file = self.cfg.folder / "detector.pt"
        logger.info(f"Saving detector to {out_file}")
        detector.save(out_file)
        return detector

    # ----- DATASETS ----- #

    def get_train_dataset(self) -> DialogueDataset | TargetedDialogueDataset:
        train_dataset = self.get_dataset(
            self.cfg.train_data, model=self.train_policy_model
        )
        num_val = int(len(train_dataset) * self.cfg.val_fraction)
        return train_dataset[:-num_val] if num_val > 0 else train_dataset

    def get_eval_datasets(self) -> list[DialogueDataset | TargetedDialogueDataset]:
        eval_datasets = [
            self.get_dataset(name, model=self.eval_policy_model)
            for name in self.cfg.eval_data
        ]

        # Add validation split if val_fraction > 0
        train_dataset = self.get_dataset(
            self.cfg.train_data, model=self.train_policy_model
        )
        num_val = int(len(train_dataset) * self.cfg.val_fraction)
        val_dataset = train_dataset[-num_val:] if num_val > 0 else None
        if val_dataset:
            assert val_dataset.variant is not None
            val_dataset.variant = val_dataset.variant + "_val"
            eval_datasets = [val_dataset] + eval_datasets

        return eval_datasets

    def get_control_datasets(self) -> list[DialogueDataset | TargetedDialogueDataset]:
        """Get control datasets using either on-policy or pre-written data based on config."""
        model = (
            get_model_name(self.cfg.model_name)
            if self.cfg.control_on_policy
            else "prewritten"
        )

        control_datasets = [
            self.get_dataset(name, model=model)[: self.cfg.control_dataset_size]
            for name in self.cfg.control_data
        ]
        return control_datasets

    @property
    def train_policy_model(self) -> str:
        """
        Get the policy model used for training the detector. Either the same model or prewritten.
        """
        if self.cfg.train_on_policy:
            return self.cfg.model_name.value
        else:
            return "prewritten"

    @property
    def eval_policy_model(self) -> str:
        """
        Get the policy model used for evaluation. Either the same model or prewritten.
        """
        if self.cfg.eval_on_policy:
            return self.cfg.model_name.value
        else:
            return "prewritten"

    def get_dataset(
        self, name: str, model: str
    ) -> DialogueDataset | TargetedDialogueDataset:
        dataset = REPOSITORY.get(
            name,
            model=model,
            trim_reasoning=self.cfg.trim_reasoning,
            shuffle_upon_init=not name.startswith("repe"),
            max_llama_token_length=self.cfg.max_llama_token_length,
        )
        return dataset

    # ---- SCORES ----- #

    def get_control_scores(self, all_tokens: bool = False) -> dict[str, Scores]:
        """
        Returns scores for each control dataset.

        Args:
            all_tokens: If True, returns scores for each prompt in the control dataset.
                        If False, returns aggregated scores for each dataset.

        Returns:
            A dict of (dataset name, scores)
        """
        file_name = (
            "control_scores_all_tokens.json" if all_tokens else "control_scores.json"
        )
        path = self.cfg.folder / file_name

        if not path.exists():
            logger.info("Computing control scores")
            self.compute_control_scores()

        with open(path) as f:
            raw_scores = json.load(f)

        return {
            dataset: Scores.from_dict(scores) for dataset, scores in raw_scores.items()
        }

    def get_eval_scores(self, all_tokens: bool = False) -> dict[str, Scores]:
        """
        Get the scores on each dataset. Returns a dict of (dataset name, PairedScores).
        If all_tokens is True, returns scores for every token.
        """
        file_name = "scores_all_tokens.json" if all_tokens else "scores.json"
        path = self.cfg.folder / file_name
        if not path.exists():
            logger.info("Computing scores")
            self.compute_eval_scores()
        with open(path) as f:
            raw_scores = json.load(f)
        for name, scores in raw_scores.items():
            assert len(scores["scores"]) == len(scores["labels"]), (
                f"Mismatch between labels (len {len(scores['labels'])}) and scores"
                f"(len {len(scores['scores'])}) for dataset {name}"
            )
        return {name: Scores.from_dict(scores) for name, scores in raw_scores.items()}

    def compute_control_scores(self) -> None:
        """Compute scores for control datasets"""
        self.compute_scores(is_control=True)

    def compute_eval_scores(self) -> None:
        """Compute scores for eval datasets"""
        self.compute_scores(is_control=False)

    def compute_scores(self, is_control: bool = False) -> None:
        """
        Compute the scores for datasets both on the detection mask and on all tokens and save
        them to file.

        If is_control is True, compute for control datasets, otherwise for eval datasets.
        """
        datasets = self.control_datasets if is_control else self.eval_datasets
        prefix = "control_" if is_control else ""

        shard_sizes = {
            "lmsys": 2,
            "sandbagging_v2__wmdp_mmlu": 2,
            "ai_audit__reasoning": 1,
            "ai_audit__no_reasoning": 1,
            "roleplaying__plain": 1,
            "insider_trading__onpolicy": 1,
            "insider_trading_doubledown__onpolicy": 1,
            "insider_trading__upscale": 2,
            "insider_trading_doubledown__upscale": 2,
            "alpaca__plain": 4,
        }

        all_scores = {}
        all_scores_all_tokens = {}
        for dataset in tqdm(
            datasets, desc=f"{'Control' if is_control else 'Eval'} Datasets"
        ):
            shard_size = shard_sizes.get(dataset.name, 12)
            scores, scores_all_tokens = self.compute_scores_for_dataset(
                dataset, shard_size
            )
            all_scores[dataset.name] = scores
            all_scores_all_tokens[dataset.name] = scores_all_tokens
            assert len(dataset.labels) == len(scores) == len(scores_all_tokens), (
                f"Mismatch labels (len {len(dataset.labels)}) and scores (len {len(scores)})"
                f" and scores_all_tokens (len {len(scores_all_tokens)}) for dataset {dataset.name}"
            )
            all_scores[dataset.name].labels = dataset.labels
            all_scores_all_tokens[dataset.name].labels = dataset.labels

        with open(self.cfg.folder / f"{prefix}scores.json", "w") as f:
            json.dump(all_scores, f, default=lambda scores: scores.to_dict())
            logger.info(f"Saved {prefix}scores to {f.name}")

        with open(self.cfg.folder / f"{prefix}scores_all_tokens.json", "w") as f:
            json.dump(all_scores_all_tokens, f, default=lambda scores: scores.to_dict())
            logger.info(f"Saved {prefix}scores_all_tokens to {f.name}")

    def compute_scores_for_dataset(
        self, dataset: DialogueDataset | TargetedDialogueDataset, shard_size: int | None
    ) -> tuple[Scores, Scores]:
        """
        Compute the scores for a dataset, return scores on the detection mask and on all tokens.

        A shard size of None means computing all acts in one shard
        """
        assert len(dataset) > 0, f"Dataset {dataset.name} is empty"

        def get_scores_for_shard(
            dataset: DialogueDataset | TargetedDialogueDataset,
        ) -> tuple[Scores, Scores]:
            toks = TokenizedDataset.from_dataset(
                dataset,
                self.tokenizer,
                detect_only_start_of_turn=self.cfg.detect_only_start_of_turn,
                detect_only_last_token=self.cfg.detect_only_last_token,
            )
            acts = self.get_acts(toks)
            detector = self.get_detector()
            scores = detector.score(acts)
            scores_all_tokens = detector.score(acts, all_acts=True)
            assert len(scores) == len(scores_all_tokens) == len(dataset) == len(toks), (
                f"Len mismatch: scores={len(scores)}, scores_all_tokens={len(scores_all_tokens)},"
                f"dataset={len(dataset)}, toks={len(toks)}"
            )
            return scores, scores_all_tokens

        all_scores: list[Scores] = []
        all_scores_all_tokens: list[Scores] = []
        shard_size = shard_size or len(dataset)
        for start_idx in trange(0, len(dataset), shard_size, desc=dataset.name):
            shard_dataset = dataset[start_idx : start_idx + shard_size]
            scores, scores_all_tokens = get_scores_for_shard(shard_dataset)
            all_scores.append(scores)
            all_scores_all_tokens.append(scores_all_tokens)

        scores = Scores.from_list(all_scores)
        scores_all_tokens = Scores.from_list(all_scores_all_tokens)
        assert len(scores) == len(scores_all_tokens) == len(dataset), (
            f"Len mismatch: scores={len(scores)}, scores_all_tokens={len(scores_all_tokens)}, "
            f"dataset={len(dataset)}"
        )
        return scores, scores_all_tokens

    def get_acts(self, toks: TokenizedDataset) -> Activations:
        acts = Activations.from_model(
            self.model, toks, batch_size=4, layers=self.cfg.detect_layers
        )
        return acts

    def recompute_scores_on_dataset(
        self, dataset_name: str, shard_size: int = 12
    ) -> None:
        """Recompute scores on a dataset (or add new dataset)"""
        dataset = self.get_dataset(dataset_name, model=self.eval_policy_model)
        scores = self.get_eval_scores(all_tokens=False)
        scores_all_tokens = self.get_eval_scores(all_tokens=True)

        dataset_scores, dataset_scores_all_tokens = self.compute_scores_for_dataset(
            dataset, shard_size=shard_size
        )
        scores[dataset_name] = dataset_scores
        scores_all_tokens[dataset_name] = dataset_scores_all_tokens
        scores[dataset.name].labels = dataset.labels
        scores_all_tokens[dataset.name].labels = dataset.labels

        with open(self.cfg.folder / "scores.json", "w") as f:
            json.dump(scores, f, default=lambda scores: scores.to_dict())
        with open(self.cfg.folder / "scores_all_tokens.json", "w") as f:
            json.dump(scores_all_tokens, f, default=lambda scores: scores.to_dict())

        # update the config
        if dataset_name not in self.cfg.eval_data:
            self.cfg.eval_data.append(dataset_name)
            self.cfg._save_config()  # type: ignore

    # ----- OTHER ----- #

    def __str__(self) -> str:
        return f"Experiment({self.cfg.name})"

    @property
    def model(self) -> PreTrainedModel:
        """Lazily load the model when first accessed."""
        if self._model is None:
            model, _ = get_model_and_tokenizer(
                self.cfg.model_name,
                cut_at_layer=self.cfg.detect_layers[-1],
            )
            self._model = model
        assert self._model is not None, "Model not initialized"
        return self._model
