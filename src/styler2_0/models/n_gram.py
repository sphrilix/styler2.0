from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import Embedding
from torch.nn.functional import relu
from torch.optim import Optimizer

from styler2_0.models.model_base import ModelBase
from styler2_0.preprocessing.model_tokenizer import (
    ModelTokenizer,
    SplitByCheckstyleTokenizer,
    SplitByTokenizer,
)
from styler2_0.utils.vocab import Vocabulary


class NGram(ModelBase):
    """
    N-Gram Model.
    Based on: https://surfertas.github.io/deeplearning/pytorch/2017/08/20/n-gram.html
    """

    def __init__(
        self,
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
        save: Path,
        input_length: int,
        output_length: int,
        device: str,
        emb_dim: int,
        linear_dim: int,
    ) -> None:
        super().__init__(
            input_length, output_length, src_vocab, trg_vocab, save, device
        )
        self.embedding = Embedding(len(src_vocab), emb_dim)
        self.linear1 = nn.Linear(emb_dim * input_length, linear_dim)
        self.linear2 = nn.Linear(linear_dim, len(trg_vocab))

    def _fit_one_batch(
        self, batch: Tensor, criterion: nn.Module, optimizer: Optimizer
    ) -> float:
        """
        Fits one batch.
        :param batch: Batch to fit.
        :param criterion: The loss function.
        :param optimizer: The optimizer.
        :return: Returns the loss.
        """
        optimizer.zero_grad()
        src = batch[0].to(self.device)
        trg = batch[1].to(self.device).squeeze(1)
        out = self(src)
        loss = criterion(out, trg)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _eval_one_batch(self, batch: Tensor, criterion: nn.Module) -> float:
        """
        Evaluates one batch.
        :param batch: The batch to evaluate.
        :param criterion: The loss function.
        :return: Returns the loss.
        """
        src = batch[0].to(self.device)
        trg = batch[1].to(self.device).squeeze(1)
        out = self(src)
        loss = criterion(out, trg)
        return loss.item()

    def forward(self, src: Tensor) -> Tensor:
        """
        Forward pass.
        :param src: Tensor to be forwarded.
        :return: Returns the output tensor.
        """
        batch_size = src.shape[0]

        # src = [batch_size, input_length]
        # embedding = [batch_size, input_length * emb_dim]
        embedding = self.embedding(src).view((batch_size, -1))

        # out = [batch_size, linear_dim]
        out = self.linear1(embedding)

        out = relu(out)

        # out = [batch_size, len(trg_vocab)]
        out = self.linear2(out)

        return out

    def _fix(self, src: Tensor, top_k: int) -> list[(float, Tensor)]:
        """
        Fix one batch.
        :param src: The source tensor.
        :param top_k: How many predictions to return.
        :return: Returns predictions with confidences.
        """
        src = src.to(self.device).unsqueeze(0)
        out = self(src)
        predictions = torch.topk(out, top_k, dim=1).indices.T
        confidences = torch.topk(out, top_k, dim=1).values.T  # noqa
        return list(zip(confidences, predictions, strict=True))

    @classmethod
    def _build_from_config(
        cls,
        params: dict[str, ...],
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
        save: Path,
    ) -> "ModelBase":
        return NGram(
            src_vocab,
            trg_vocab,
            save,
            params["input_length"],
            params["output_length"],
            params["device"],
            params["emb_dim"],
            params["linear_dim"],
        )

    def _inp_tokenizer(self) -> ModelTokenizer:
        return SplitByCheckstyleTokenizer()

    def _post_process_fixes(self, fixes: list[list[str]]) -> list[list[str]]:
        tokenizer = SplitByTokenizer(self.input_length)
        return super()._post_process_fixes(
            [tokenizer.get_tokens(fix[0]) for fix in fixes]
        )
