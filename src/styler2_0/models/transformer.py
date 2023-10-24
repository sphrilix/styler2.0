import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from src.styler2_0.models.model_base import ModelBase
from src.styler2_0.utils.vocab import Vocabulary


class Transformer(ModelBase):
    """
    Transformer for translation.
    Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    def __init__(
        self,
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
        save: Path,
        device: str,
        input_length: int,
        output_length: int,
        emb_size: int,
        n_head: int,
        n_layers_enc: int,
        n_layers_dec: int,
        dropout: float,
        dim_feedforward: int,
    ) -> None:
        super().__init__(
            input_length, output_length, src_vocab, trg_vocab, save, device
        )
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=n_head,
            num_encoder_layers=n_layers_enc,
            num_decoder_layers=n_layers_dec,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, len(self.trg_vocab))
        self.src_tok_emb = TokenEmbedding(len(self.src_vocab), emb_size)
        self.tgt_tok_emb = TokenEmbedding(len(self.trg_vocab), emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout, max_len=self.input_length
        )

    def _fit_one_batch(
        self, batch: Tensor, criterion: nn.Module, optimizer: Optimizer
    ) -> float:
        src = batch[0].T
        trg = batch[1].T
        trg_inp = trg[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self._create_mask(
            src, trg_inp
        )
        optimizer.zero_grad()
        output = self(
            src,
            trg_inp,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )
        trg_out = trg[1:, :]
        optimizer.zero_grad()
        loss = criterion(output.reshape(-1, output.shape[-1]), trg_out.reshape(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

    def _eval_one_batch(self, batch: Tensor, criterion: nn.Module) -> float:
        src = batch[0].T
        trg = batch[1].T
        trg_inp = trg[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self._create_mask(
            src, trg_inp
        )
        output = self(
            src,
            trg_inp,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )
        trg_out = trg[1:, :]
        loss = criterion(output.reshape(-1, output.shape[-1]), trg_out.reshape(-1))
        return loss.item()

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ) -> Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def _encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def _decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )

    def _fix(self, src: Tensor, top_k: int) -> list[(float, Tensor)]:
        src = src.unsqueeze(0).T
        num_tokens = src.shape[0]
        src_mask = (
            (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(self.device)
        )
        memory = self._encode(src, src_mask)
        ys = (
            torch.ones(1, 1)
            .fill_(self.trg_vocab[self.trg_vocab.sos])
            .type(torch.long)
            .to(self.device)
        )
        for _ in range(self.output_length - 1):
            memory = memory.to(self.device)
            tgt_mask = (
                self._generate_square_subsequent_mask(ys.size(0)).type(torch.bool)
            ).to(self.device)
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0
            )
        return [(0.0, ys)]

    @classmethod
    def _build_from_config(
        cls,
        params: dict[str, ...],
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
        save: Path,
    ) -> "Transformer":
        return Transformer(
            src_vocab,
            trg_vocab,
            save,
            params["device"],
            params["input_length"],
            params["output_length"],
            params["emb_size"],
            params["n_head"],
            params["n_layers"],
            params["n_layers"],
            params["dropout"],
            params["dim_feed_forward"],
        )

    def _generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(
            0, 1
        )
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _create_mask(
        self, src: Tensor, tgt: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(
            torch.bool
        )

        src_padding_mask = (
            (src == self.src_vocab[self.src_vocab.pad]).transpose(0, 1).to(self.device)
        )
        tgt_padding_mask = (
            (tgt == self.src_vocab[self.trg_vocab.pad]).transpose(0, 1).to(self.device)
        )
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer.
    From https://pytorch.org/tutorials/beginner/translation_transformer.html
    """

    def __init__(self, emb_size: int, dropout: float, max_len: int) -> None:
        """
        Initialize Positional Encoding.
        :param emb_size: The embedding size.
        :param dropout: Fraction of the embeddings to drop.
        :param max_len: Maximum length of the sequence.
        """
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        """
        Forward pass of Positional Encoding.
        :param token_embedding: The token embedding.
        :return: Returns the token embedding with positional encoding.
        """
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    """
    Helper for token embedding.
    From: https://pytorch.org/tutorials/beginner/translation_transformer.html
    """

    def __init__(self, vocab_size: int, emb_size: int) -> None:
        """
        Initialize Token Embedding.
        :param vocab_size: The input vocabulary size.
        :param emb_size: The embedding size.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Forward pass of Token Embedding.
        The embedding is scaled by sqrt(emb_size), because of how
        transformers are trained in pytorch.
        :param tokens: The input tokens.
        :return: Returns the embedded token sequence.
        """
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
