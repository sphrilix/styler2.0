import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from src.styler2_0.models.model_base import BeamSearchDecodingStepData, ModelBase
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
        """
        Fit one batch.
        :param batch: The input batch = [src, trg]
        :param criterion: The loss function to be used.
        :param optimizer: The optimizer to be used.
        :return: Returns the loss.
        """

        # batch = [src, trg]
        # trg, src = [batch_size, src_len]
        # but model requires [src_len, batch_size]
        src = batch[0].T
        trg = batch[1].T

        # shift right
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
        """
        Evaluate one batch.
        :param batch: The batch to evaluate [src, trg]
        :param criterion: The loss function to be used.
        :return: Returns the loss of the batch.
        """
        # src, trg = [batch_size, src_len]
        # but the models requires [src_len, batch_size]
        src = batch[0].T
        trg = batch[1].T

        # Shift trg right
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
        """
        Forward pass of the transformer.
        :param src: The input sequence [src_len, batch_size]
        :param trg: The output sequence [trg_len, batch_size] (shifted right)
        :param src_mask: The input mask
        :param tgt_mask: The output mask
        :param src_padding_mask: The input padding mask
        :param tgt_padding_mask: The output padding mask
        :param memory_key_padding_mask: The memory padding mask
        :return: Returns the output tensor.
        """
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
        """
        Apply encoding to the input sequence.
        :param src: The input sequence [src_len, batch_size]
        :param src_mask: The input mask
        :return: Returns the encoded input sequence.
        """
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def _decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        """
        Apply decoding to the input sequence.
        :param tgt: The input sequence [tgt_len, batch_size]
        :param memory: The encoder output
        :param tgt_mask: The trg mask
        :return: Returns the decoded input sequence.
        """
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )

    def _fix(self, src: Tensor, top_k: int) -> list[(float, Tensor)]:
        """
        Fix the input sequence and return the top_k decoded sequences
        using Beam Search decoding.
        :param src: The input sequence [src_len]
        :param top_k: How many fixes should be sampled.
        :return: Returns #top_k decoded sequences.
        """
        src = src.unsqueeze(0).T
        return self._beam_search(src, top_k)

    def _beam_search(self, src: Tensor, beam_width: int) -> list[(float, Tensor)]:
        """
        Beam search decoding for the Transformer model.
        Currently, only working with batch_size == 1!
        :param src: The input sequence [src_len, 1]
        :param beam_width: The beam width.
        :return: Returns #beam_width decoded sequences.
        """
        assert src.shape[1] == 1, "Currently, working only with batch_size == 1"

        # create mask for masking <UNK> to negative infinity
        unk_mask = (
            torch.zeros(1, len(self.trg_vocab))
            .index_fill_(
                1, torch.tensor([self.trg_vocab.stoi(self.trg_vocab.unk)]), -torch.inf
            )
            .to(self.device)
        )

        num_tokens = src.shape[0]

        # Create the source mask
        src_mask = (
            (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(self.device)
        )

        # Encode the source sequence
        memory = self._encode(src, src_mask)

        # Init with one as in the first step only #beam_width can be sampled
        # and then #beam_width^2 and <SOS> is always the first token
        search_space = [
            BeamSearchDecodingStepData(
                torch.tensor([self.trg_vocab.stoi(self.trg_vocab.sos)]).to(self.device),
                {"memory": memory},
                1.0,  # <SOS> is always the first token
                self.trg_vocab.stoi(self.trg_vocab.eos),
            )
        ]
        for _ in range(1, self.output_length):
            # Init new search space as set to eliminate duplicates
            # TODO: check if really necessary as those may only arose due to
            #       not stopping the search when <EOS> is reached
            new_search_space = set()
            for sample in search_space:
                # If the sequence is finished, add it to the new search space
                # and continue,as we don't need to predict anything for this
                # sequence anymore
                if sample.is_sequence_finished():
                    new_search_space.add(sample)
                    continue

                # Fit sequence into the transformer model.
                inp = sample.sequence.unsqueeze(dim=1)
                tgt_mask = (
                    self._generate_square_subsequent_mask(inp.size(0)).type(torch.bool)
                ).to(self.device)
                out = self._decode(inp, sample.state["memory"], tgt_mask)
                out = out.transpose(0, 1)

                # Get the last prediction and apply the unk_mask
                out = self.generator(out[:, -1])
                out = out + unk_mask

                # Apply softmax to get the probabilities in [0,1]
                out = nn.Softmax(dim=1)(out)
                confidences, indices = out.topk(beam_width, dim=1)
                # Iterate over top k predictions and add them to the new search space
                for conf, index in zip(
                    confidences.squeeze(0), indices.squeeze(0), strict=True
                ):
                    new_search_space.add(
                        BeamSearchDecodingStepData(
                            torch.cat((sample.sequence, index.unsqueeze(0)), dim=0),
                            {"memory": memory},
                            sample.confidence * conf.item(),  # Calculate new confidence
                            sample.end_token_idx,
                        )
                    )
            # Sort the new search space by confidence and take the top
            # #beam_width samples
            search_space = sorted(new_search_space, reverse=True)[:beam_width]

        # Return the top #beam_width samples and their confidences
        return [(sample.confidence, sample.sequence) for sample in search_space]

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

    def _generate_square_subsequent_mask(self, sz: int | tuple[int, ...]) -> Tensor:
        """
        Generate a square mask for the sequence. The masked positions are filled.
        This is done to prevent the model from cheating by looking into the future.
        :param sz: The size of the mask.
        :return: Returns the mask [sz, sz].
        """
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
        """
        Create masks for src and tgt
        :param src: The source tensor.
        :param tgt: The target tensor.
        :return: Returns the masks for src, tgt and the corresponding padding masks.
        """
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        # prevent model for looking into the future.
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(
            torch.bool
        )

        # Mask the padding tokens.
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
    This is done to introduce a notion of order into the sequence.
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
