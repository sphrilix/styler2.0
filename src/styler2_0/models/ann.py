from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import Dropout, Embedding, Linear, Parameter
from torch.optim import Optimizer

from styler2_0.models.model_base import ModelBase
from styler2_0.preprocessing.model_tokenizer import ModelTokenizer
from styler2_0.utils.vocab import Vocabulary


class ANN(ModelBase):
    """
    Attentional Neural Network similar to code2vec.
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
        self.embedding = Embedding(input_length, emb_dim)
        self.W = Parameter(torch.randn((1, emb_dim, emb_dim)))
        self.a = Parameter(torch.randn((1, emb_dim, 1)))
        self.out = Linear(emb_dim, output_length)
        self.dropout = Dropout(dropout)

        super().__init__(
            input_length, output_length, src_vocab, trg_vocab, save, device
        )

    def _fit_one_batch(
        self, batch: Tensor, criterion: nn.Module, optimizer: Optimizer
    ) -> float:
        src = batch[0]
        trg = batch[1]
        optimizer.zero_grad()
        out = self(src)
        loss = criterion(out, trg)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _eval_one_batch(self, batch: Tensor, criterion: nn.Module) -> float:
        src = batch[0]
        trg = batch[1]
        out = self(src)
        loss = criterion(out, trg)
        return loss.item()

    def forward(self, src: Tensor) -> Tensor:
        batch_size = src.shape[0]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded, dim=2)
        W = self.W.repeat(batch_size, 1, 1)
        c = embedded.permute(0, 2, 1)
        x = torch.tanh(torch.bmm(W, c))
        x = x.permute(0, 2, 1)
        a = self.a.repeat(batch_size, 1, 1)
        z = torch.bmm(x, a).squeeze(2)
        z = torch.softmax(z, dim=1)
        z = z.unsqueeze(2)
        v = torch.bmm(x, z).squeeze(2)
        return self.out(v)

    def _fix(self, src: Tensor, top_k: int) -> list[(float, Tensor)]:
        pass

    @classmethod
    def _build_from_config(
        cls,
        params: dict[str, ...],
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
        save: Path,
    ) -> "ModelBase":
        pass

    def _inp_tokenizer(self) -> ModelTokenizer:
        pass
