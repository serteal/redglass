from typing import Dict

from torch.utils.data import DataLoader

from redglass.detectors import Detector
from redglass.models import Model


def train_detector(
    detector: Detector,
    model: Model,
    train_loader: DataLoader,
    val_loaders: Dict[str, DataLoader],
):
    pass
