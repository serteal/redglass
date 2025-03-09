from pathlib import Path

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from redglass.activations import Activations, PairedActivations
from redglass.data import CircuitBreakersDataset, JailbreakBenchDataset
from redglass.data.tokenized_data import TokenizedDataset
from redglass.detectors import (
    CovarianceAdjustedMMSDetector,
    DirectionDetector,
    LATDetector,
    LogisticRegressionDetector,
    MLPDetector,
    MMSDetector,
)
from redglass.metrics import get_auroc


@pytest.fixture(scope="module")
def detector_test_data(
    gemma_model: PreTrainedModel, gemma_tokenizer: PreTrainedTokenizerBase
):
    fit_data = JailbreakBenchDataset(variant="plain", shuffle_upon_init=False)
    n_train = int(len(fit_data) * 0.7)
    train_data, test_data = fit_data[:n_train], fit_data[n_train:]

    train_split_acts = PairedActivations.from_dataset(
        train_data, gemma_tokenizer, gemma_model, enforce_equal_detect_lengths=True
    )
    test_split_acts = PairedActivations.from_dataset(
        test_data, gemma_tokenizer, gemma_model, enforce_equal_detect_lengths=True
    )

    safe_train_acts, unsafe_train_acts = train_split_acts.get_all_acts()
    safe_test_acts, unsafe_test_acts = test_split_acts.get_all_acts()

    return {
        "safe_train_acts": safe_train_acts,
        "unsafe_train_acts": unsafe_train_acts,
        "safe_test_acts": safe_test_acts,
        "unsafe_test_acts": unsafe_test_acts,
    }


@pytest.mark.gpu
@pytest.mark.parametrize(
    "detector_class",
    [
        MMSDetector,
        LATDetector,
        LogisticRegressionDetector,
        MLPDetector,
        CovarianceAdjustedMMSDetector,
    ],
)
def test_test_classification_accuracy(
    detector_test_data: dict[str, Activations],
    detector_class: type[DirectionDetector],
):
    """Takes the dataset we use to train detectors, splits into train and test, and ensures we get
    high accuracy on the test set when fitting to the train set."""

    detector = detector_class(layers=list(range(4, 12)))
    detector.fit(
        detector_test_data["safe_train_acts"],
        detector_test_data["unsafe_train_acts"],
    )

    test_safe_scores = detector.score(detector_test_data["safe_test_acts"])
    test_unsafe_scores = detector.score(detector_test_data["unsafe_test_acts"])

    assert not any([score_row.isnan().any() for score_row in test_safe_scores.scores])
    assert not any([score_row.isnan().any() for score_row in test_unsafe_scores.scores])

    mean_safe_scores = torch.tensor(
        [score.mean() for score in test_safe_scores.scores if len(score) > 0]
    )
    mean_unsafe_scores = torch.tensor(
        [score.mean() for score in test_unsafe_scores.scores if len(score) > 0]
    )

    accuracy = (mean_safe_scores > mean_unsafe_scores).float().mean()
    assert accuracy > 0.99, f"Accuracy is only {accuracy}"


@pytest.mark.gpu
def test_generalization(
    gemma_model: PreTrainedModel,
    gemma_tokenizer: PreTrainedTokenizerBase,
):
    # Prepare CircuitBreakersDataset (train set)
    train_data = CircuitBreakersDataset()
    split_acts = PairedActivations.from_dataset(
        train_data, gemma_tokenizer, gemma_model, layers=list(range(9, 15))
    )
    safe_train_acts = split_acts.safe_acts
    unsafe_train_acts = split_acts.unsafe_acts

    # Prepare JailbreakBenchDataset (test set)
    test_data = JailbreakBenchDataset()
    split_test_acts = PairedActivations.from_dataset(
        test_data, gemma_tokenizer, gemma_model, layers=list(range(9, 15))
    )
    safe_test_acts = split_test_acts.safe_acts
    unsafe_test_acts = split_test_acts.unsafe_acts

    # Train and evaluate detector
    detector = MMSDetector(layers=list(range(9, 15)))
    detector.fit(unsafe_train_acts, safe_train_acts)

    test_safe_scores = detector.score(safe_test_acts)
    mean_safe_scores = torch.tensor([score.mean() for score in test_safe_scores.scores])
    test_unsafe_scores = detector.score(unsafe_test_acts)
    mean_unsafe_scores = torch.tensor(
        [score.mean() for score in test_unsafe_scores.scores]
    )

    auroc = get_auroc(mean_safe_scores, mean_unsafe_scores)

    assert auroc > 0.95, f"AUROC is only {auroc}"


@pytest.mark.parametrize(
    "detector_class", [MMSDetector, LATDetector, LogisticRegressionDetector]
)
def test_save_load_detector(tmp_path: Path, detector_class: type[DirectionDetector]):
    # Create a detector with random directions
    detector = detector_class(layers=[0, 1, 2])
    detector.directions = torch.rand(3, 768)  # Assuming 768 is the embedding size

    # Save the detector
    save_path = tmp_path / "test_detector.pkl"
    detector.save(save_path)

    # Load the detector
    loaded_detector = detector_class.load(save_path)

    # Check if the loaded detector has the same attributes
    assert detector.layers == loaded_detector.layers
    assert loaded_detector.directions is not None

    # Ensure both tensors are on the same device before comparison
    assert detector.directions.device == loaded_detector.directions.device
    assert torch.allclose(detector.directions, loaded_detector.directions)


@pytest.fixture(scope="module")
def random_test_data(
    gemma_tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Activations]:
    """Alternative to detector_test_data that doesn't need a gpu."""
    layers = [4]
    emb_size = 24
    fit_data = CircuitBreakersDataset()[:50]
    tokenized_dataset = TokenizedDataset.from_dataset(fit_data, gemma_tokenizer)

    # Ensure detection mask has at least some True values
    if (
        tokenized_dataset.detection_mask is None
        or not tokenized_dataset.detection_mask.any()
    ):
        # Create a simple detection mask with some True values if needed
        tokenized_dataset.detection_mask = torch.zeros_like(
            tokenized_dataset.attention_mask, dtype=torch.bool
        )
        tokenized_dataset.detection_mask[:, 0] = (
            True  # Mark first token in each sequence for detection
        )

    def get_acts() -> Activations:
        shape = tokenized_dataset.tokens.shape + (len(layers), emb_size)
        return Activations(
            all_acts=torch.randn(shape),
            tokenized_dataset=tokenized_dataset,
            layers=layers,
        )

    acts = get_acts()
    return {
        "safe_train_acts": acts,
        "unsafe_train_acts": acts,
        "safe_test_acts": acts,
        "unsafe_test_acts": acts,
    }


@pytest.mark.parametrize("detector_class", [MMSDetector, LogisticRegressionDetector])
def test_fit_detector_with_random_acts(
    detector_class: type[DirectionDetector],
    random_test_data: dict[str, Activations],
):
    detector = detector_class(layers=[4])
    detector.fit(
        random_test_data["safe_train_acts"], random_test_data["unsafe_train_acts"]
    )
    assert detector.directions is not None
    emb_size = random_test_data["safe_train_acts"].all_acts.shape[-1]
    assert detector.directions.shape == (len(detector.layers), emb_size)

    detector.score(random_test_data["safe_test_acts"])


@pytest.mark.parametrize("detector_class", [MMSDetector, LogisticRegressionDetector])
def test_detector_with_sparse_activations(
    random_test_data: dict[str, Activations], detector_class: type[DirectionDetector]
):
    """Tests that detectors can score activations with sparse tensors"""

    # Train detector on dense activations
    if detector_class == LogisticRegressionDetector:
        detector = LogisticRegressionDetector(layers=[4], normalize=False)
    else:
        detector = detector_class(layers=[4])
    detector.fit(
        random_test_data["safe_train_acts"], random_test_data["unsafe_train_acts"]
    )

    # Save test activations in sparse format
    dense_acts = random_test_data["safe_test_acts"]
    sparse_acts = Activations(
        all_acts=dense_acts.all_acts.to_sparse_coo(),
        tokenized_dataset=dense_acts.tokenized_dataset,
        layers=dense_acts.layers,
    )

    # Ensure sparse format is correct before scoring
    if sparse_acts.all_acts.is_sparse:
        assert (
            sparse_acts.all_acts.sparse_dim() + sparse_acts.all_acts.dense_dim()
            == len(sparse_acts.all_acts.shape)
        )

    # Score using sparse activations
    sparse_scores = detector.score(sparse_acts)
    dense_scores = detector.score(dense_acts)

    # Compare scores between sparse and dense versions
    for sparse_score, dense_score in zip(
        sparse_scores.scores, dense_scores.scores, strict=True
    ):
        assert torch.allclose(sparse_score, dense_score, atol=1e-5)
