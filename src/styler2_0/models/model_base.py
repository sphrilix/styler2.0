import math
import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor, long, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.styler2_0.utils.tokenize import ProcessedSourceFile
from src.styler2_0.utils.vocab import Vocabulary


class ModelBase(nn.Module, ABC):
    """
    Abstract base class for a model.
    """

    CURR_DIR = Path(os.path.dirname(os.path.relpath(__file__)))
    CONFIGS_PATH = CURR_DIR / Path("../../../config/models/")
    SAVE_PATH = CURR_DIR / Path("../../../models/")

    def __init__(
        self,
        input_length: int,
        output_length: int,
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
        save: Path = None,
    ) -> None:
        self.input_length = input_length
        self.output_length = output_length
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        if not save:
            self.save = self.SAVE_PATH / Path(self.__class__.__name__)
        os.makedirs(self.save, exist_ok=True)
        super().__init__()

    def _fit_one_epoch(
        self, data: DataLoader, criterion: nn.Module, optimizer: Optimizer
    ) -> float:
        """
        Fit one epoch.
        :param data: The data to fit.
        :param criterion: The loss function.
        :param optimizer: The optimizer.
        :return: The loss.
        """
        self.train()
        epoch_loss = 0
        for batch in data:
            epoch_loss += self._fit_one_batch(batch, criterion, optimizer)
        return epoch_loss / len(data)

    def _eval_one_epoch(self, data: DataLoader, criterion: nn.Module) -> float:
        """
        Evaluate one epoch.
        :param data: The data to evaluate.
        :param criterion: The loss function.
        :return: The loss.
        """
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
        """
        Fit one batch.
        :param batch: The batch to be fitted.
        :param criterion: The criterion to be used.
        :param optimizer: The optimizer to be used.
        :return: The loss.
        """
        pass

    @abstractmethod
    def _eval_one_batch(self, batch: Tensor, criterion: nn.Module) -> float:
        """
        Evaluate one batch.
        :param batch: The batch to be evaluated.
        :param criterion: The criterion to be used.
        :return: The loss.
        """
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

            torch.save(self.state_dict(), self.save / f"model_{epoch}_{valid_loss}.pt")

    @abstractmethod
    def forward(self, *args: ..., **kwargs: ...) -> Tensor:
        """
        Forward pass.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        :return: The output.
        """
        pass

    def fix(self, src: ProcessedSourceFile, top_k: int) -> list[ProcessedSourceFile]:
        """
        Fix the given source file.
        Currently, only files with one violation are supported. (same as styler)
        :param src: The source file to be fixed.
        :param top_k: How many samples are searched to find the shortest fix.
        :return: The fixed source file.
        """
        assert len(src.report.violations) == 1
        affected_tokens = list(src.violations_with_ctx())[0]
        input_ids = [self.src_vocab.stoi(str(t)) for t in affected_tokens]
        src_tenor = self._process_predict_input(input_ids)
        possible_fixes = self._fix(src_tenor, top_k)
        return list(
            src.get_fixes_for(
                possible_fixes, (src.checkstyle_tokens[0], src.checkstyle_tokens[-1])
            )
        )

    @abstractmethod
    def _fix(self, src: Tensor, top_k: int) -> list[(float, Tensor)]:
        """
        Get the predictions from the model to fix the violation.
        :param src: The input tensor.
        :param top_k: How many samples are searched to find the shortest fix.
        :return: Returns the output Tensors and the confidences.
        """
        pass

    def _process_predict_input(self, inp: list[int]) -> Tensor:
        """
        Process the input for the predict method.
        :param inp: The input.
        :return: The processed input.
        """
        inp = (
            [self.src_vocab[self.src_vocab.sos]]
            + inp[: self.input_length - 1]
            + [self.src_vocab[self.src_vocab.eos]]
        )
        inp = inp + [self.src_vocab[self.src_vocab.pad]] * (
            self.input_length - len(inp)
        )
        return torch.tensor(inp, dtype=long, device=self.device)
