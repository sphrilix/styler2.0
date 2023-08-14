import math
import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class ModelBase(nn.Module, ABC):
    """
    Abstract base class for a model.
    """

    CURR_DIR = Path(os.path.dirname(os.path.relpath(__file__)))
    CONFIGS_PATH = CURR_DIR / Path("../../../config/models/")

    def __init__(self, input_length: int, output_length: int) -> None:
        self.input_length = input_length
        self.output_length = output_length
        super().__init__()

    def _fit_one_epoch(
        self, data: DataLoader, criterion: nn.Module, optimizer: Optimizer
    ) -> float:
        self.train()
        epoch_loss = 0
        for batch in data:
            epoch_loss += self._fit_one_batch(batch, criterion, optimizer)
        return epoch_loss / len(data)

    def _eval_one_epoch(self, data: DataLoader, criterion: nn.Module) -> float:
        self.eval()
        epoch_loss = 0
        with torch.no_grad():
            for batch in data:
                epoch_loss += self._eval_one_batch(batch, criterion)
        return epoch_loss / len(data)

    @abstractmethod
    def _fit_one_batch(
        self, batch: Tensor, criterion: nn.Module, optimizer: Optimizer
    ) -> float:
        pass

    @abstractmethod
    def _eval_one_batch(self, batch: Tensor, criterion: nn.Module) -> float:
        pass

    def fit(
        self,
        epochs: int,
        train_data: DataLoader,
        val_data: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
    ) -> None:
        """
        Call this method to train the model.
        :param epochs: How many epochs.
        :param train_data: Trainings data.
        :param val_data: Validation data.
        :param criterion: Loss function.
        :param optimizer: Optimization rule.
        :return:
        """
        for epoch in range(epochs):
            train_loss = self._fit_one_epoch(train_data, criterion, optimizer)
            valid_loss = self._eval_one_epoch(val_data, criterion)
            print(f"Epoch: {epoch + 1:02}")
            print(
                f"\tTrain Loss: {train_loss:.3f} "
                f"| Train PPL: {math.exp(train_loss):7.3f}"
            )
            print(
                f"\t Val. Loss: {valid_loss:.3f} "
                f"|  Val. PPL: {math.exp(valid_loss):7.3f}"
            )

    @abstractmethod
    def forward(self, *args: ..., **kwargs: ...) -> Tensor:
        pass

    @classmethod
    @abstractmethod
    def build_from_config(cls) -> "ModelBase":
        pass
