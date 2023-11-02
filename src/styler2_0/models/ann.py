from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import Dropout, Embedding, Linear, Parameter
from torch.nn.functional import softmax
from torch.optim import Optimizer

from styler2_0.models.model_base import ModelBase
from styler2_0.preprocessing.model_tokenizer import ModelTokenizer, SplitByTokenizer
from styler2_0.utils.vocab import Vocabulary


class ANN(ModelBase):
    """
    Attentional Neural Network similar to code2vec.
    Based on the paper: https://arxiv.org/pdf/1803.09473.pdf
    """

    def __init__(
        self,
        input_length: int,
        output_length: int,
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
        save: Path,
        device: str,
        emb_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(
            input_length, output_length, src_vocab, trg_vocab, save, device
        )
        self.embedding = Embedding(input_length, emb_dim)
        self.W = Parameter(torch.randn((1, emb_dim, emb_dim)))
        self.a = Parameter(torch.randn((1, emb_dim, 1)))
        self.out = Linear(emb_dim, len(trg_vocab))
        self.dropout = Dropout(dropout)

    def _fit_one_batch(
        self, batch: Tensor, criterion: nn.Module, optimizer: Optimizer
    ) -> float:
        """
        Fits one batch.
        :param batch: The batch to fit.
        :param criterion: The loss function.
        :param optimizer: The optimizer.
        :return: Returns the loss on the batch.
        """
        src = batch[0].to(self.device)

        # trg = [batch_size, 1] -> [batch_size]
        trg = batch[1].to(self.device).squeeze(1)
        optimizer.zero_grad()
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
        :return: Returns the loss on the batch.
        """
        src = batch[0].to(self.device)

        # trg = [batch_size, 1] -> [batch_size]
        trg = batch[1].T.to(self.device).squeeze(1)
        out = self(src)
        loss = criterion(out, trg)
        return loss.item()

    def forward(self, src: Tensor) -> Tensor:
        """
        Forward pass of the ANN.
        As described in the paper: https://arxiv.org/pdf/1803.09473.pdf
        :param src: The input tensor.
        :return: Returns the output logits.
        """
        batch_size = src.shape[0]

        # embedded = [batch_size, input_length, emb_dim]
        embedded = self.embedding(src).to(self.device)

        # W = [batch_size, emb_dim, emb_dim]
        W = self.W.repeat(batch_size, 1, 1)

        # c = [batch_size, emb_dim, input_length]
        c = embedded.permute(0, 2, 1)

        # x = [batch_size, emb_dim, input_length]
        x = torch.tanh(torch.bmm(W, c))

        # x = [batch_size, input_length, emb_dim]
        x = x.permute(0, 2, 1)

        # a = [batch_size, emb_dim, 1]
        a = self.a.repeat(batch_size, 1, 1)

        # z = [batch_size, input_length, 1]
        z = torch.bmm(x, a).squeeze(2)
        z = softmax(z, dim=1)

        # z = [batch_size, input_length]
        z = z.unsqueeze(2)

        # x = [batch_size, emb_dim, input_length]
        x = x.permute(0, 2, 1)

        # v = [batch_size, emb_dim]
        v = torch.bmm(x, z).squeeze(2)

        # returns [batch_size, trg_vocab_size]
        return self.out(v)

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
        return ANN(
            params["input_length"],
            params["output_length"],
            src_vocab,
            trg_vocab,
            save,
            params["device"],
            params["emb_dim"],
            params["dropout"],
        )

    def _inp_tokenizer(self) -> ModelTokenizer:
        return SplitByTokenizer(self.input_length)

    def _post_process_fixes(self, fixes: list[list[str]]) -> list[list[str]]:
        """
        Post process fixes to return sub-tokens in order to be applied.
        :param fixes: Fixes to postprocess.
        :return: Returns the corrected fixes.
        """
        return [self._inp_tokenizer().get_tokens(fix[0]) for fix in fixes]
