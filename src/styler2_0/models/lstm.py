import random
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.types import Device

from src.styler2_0.models.model_base import BeamSearchDecodingStepData, ModelBase
from src.styler2_0.utils.vocab import Vocabulary
from styler2_0.preprocessing.model_tokenizer import ModelTokenizer, SequenceTokenizer


class LSTMEncoder(nn.Module):
    def __init__(
        self, input_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float
    ) -> None:
        """
        Initialize the encoder.
        :param input_dim: size of the input vocabulary
        :param emb_dim:  size of the embedding
        :param hid_dim: size of the hidden layer
        :param n_layers: size of the number of layers
        :param dropout: percentage of dropout
        """
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(
            emb_dim, hid_dim, n_layers, bidirectional=True, dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> (Tensor, Tensor):
        """
        Forward pass of the encoder.
        :param src: input tensor
        :return: hidden and cell states
        """

        # src = [src len, batch size]
        batch_size = src.shape[1]
        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        # TODO:
        #  - reshape for more than 1 layer -> grep last outputs
        #  - check if reshape actually performs the desired outcome
        return hidden.reshape(
            self.n_layers, batch_size, self.hid_dim * 2
        ), cell.reshape(self.n_layers, batch_size, self.hid_dim * 2)


class LSTMDecoder(nn.Module):
    def __init__(
        self, output_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float
    ) -> None:
        """
        Initialize the decoder.
        :param output_dim: size of the output vocabulary
        :param emb_dim: size of the embedding
        :param hid_dim: size of the hidden layer
        :param n_layers: number of layers
        :param dropout: percentage of dropout
        """
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim * 2, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, input: Tensor, hidden: Tensor, cell: Tensor
    ) -> (Tensor, Tensor, Tensor):
        """
        Forward pass of the decoder.
        :param input: input tensor
        :param hidden: hidden state
        :param cell: cell state
        :return: prediction, hidden and cell states
        """

        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class LSTM(ModelBase):
    @classmethod
    def _build_from_config(
        cls,
        params: dict[str, ...],
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
        save: Path,
    ) -> "LSTM":
        """
        Build the model from the configuration file.
        :return: the model.
        """
        input_length = params["input_length"]
        output_length = params["output_length"]
        encoder = LSTMEncoder(
            input_length,
            params["enc_emb_dim"],
            params["hidden_dim"],
            params["n_layers"],
            params["enc_dropout"],
        )
        decoder = LSTMDecoder(
            len(trg_vocab),
            params["dec_emb_dim"],
            params["hidden_dim"],
            params["n_layers"],
            params["dec_dropout"],
        )
        return LSTM(
            encoder,
            decoder,
            params["device"],
            input_length,
            output_length,
            src_vocab,
            trg_vocab,
            save,
        )

    def __init__(
        self,
        encoder: LSTMEncoder,
        decoder: LSTMDecoder,
        device: Device,
        input_length: int,
        output_length: int,
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
        save: Path,
    ) -> None:
        """
        Initialize the model.
        :param encoder: the encoder
        :param decoder: the decoder
        :param device: the device
        :param input_length: input length
        :param output_length: output length
        """
        super().__init__(
            input_length, output_length, src_vocab, trg_vocab, save, device
        )
        self.encoder = encoder
        self.decoder = decoder

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(
        self, src: Tensor, trg: Tensor, teacher_forcing_ratio: float = 0.5
    ) -> Tensor:
        """
        Forward pass of the model.
        :param src: input tensor
        :param trg: output tensor
        :param teacher_forcing_ratio: teacher forcing ratio
        :return: predictions
        """
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth
        # inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state
        # of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        inp = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(inp, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            inp = trg[t] if teacher_force else top1

        return outputs

    def _beam_search(
        self, src: Tensor, beam_width: int = 5
    ) -> list[tuple[Tensor, float]]:
        """
        Beam search decoding step used for inference.
        Currently, working only with batch_size == 1!
        :param beam_width: The beam width
        :param src: input tensor of shape [src len, 1]
        :return: predictions
        """
        # src = [src len, 1]

        assert src.shape[1] == 1, "Currently, working only with batch_size == 1"

        trg_len = self.output_length

        # create mask for masking <UNK> to negative infinity
        unk_mask = (
            torch.zeros(1, len(self.trg_vocab))
            .index_fill_(
                1, torch.tensor([self.trg_vocab.stoi(self.trg_vocab.unk)]), -torch.inf
            )
            .to(self.device)
        )

        # last hidden state of the encoder is used as the initial hidden state
        # of the decoder
        hidden, cell = self.encoder(src)

        # Init with one as in the first step only #beam_width can be sampled
        # and then #beam_width^2 and <SOS> is always the first token
        search_space = [
            BeamSearchDecodingStepData(
                torch.tensor([self.trg_vocab.stoi(self.trg_vocab.sos)]).to(self.device),
                {"hidden": hidden, "cell": cell},
                1.0,  # <SOS> is always the first token
                self.trg_vocab.stoi(self.trg_vocab.eos),
            )
        ]

        for _ in range(1, trg_len):
            # Init new search space as set to eliminate duplicates
            # TODO: check if really necessary as those may only arose due to
            #       not stopping the search when <EOS> is reached
            new_search_space = set()

            # Iterate over all samples in the search space and get predictions for each
            for sample in search_space:
                # If the sequence is finished, add it to the new search space
                # and continue,as we don't need to predict anything for this
                # sequence anymore
                if sample.is_sequence_finished():
                    new_search_space.add(sample)
                    continue

                # squeeze to add batch dimension to fit into decoding cell
                inp = sample.sequence[-1].unsqueeze(0)
                output, hidden, cell = self.decoder(
                    inp, sample.state["hidden"], sample.state["cell"]
                )

                # Mask prob <UNK> token to negative infinity
                output = output + unk_mask

                # Apply Softmax to get confidences between [0, 1]
                output = nn.Softmax(dim=1)(output)

                confidences, indices = output.topk(beam_width, dim=1)

                # Iterate over top k predictions and add them to the new search space
                for conf, index in zip(
                    confidences.squeeze(0), indices.squeeze(0), strict=True
                ):
                    new_search_space.add(
                        BeamSearchDecodingStepData(
                            torch.cat((sample.sequence, index.unsqueeze(0)), dim=0),
                            {"hidden": hidden, "cell": cell},
                            sample.confidence * conf.item(),  # Calculate new confidence
                            sample.end_token_idx,
                        )
                    )

            # Sort the new search space by confidence and take the top
            # #beam_width samples
            search_space = sorted(new_search_space, reverse=True)[:beam_width]

        # Return the top #beam_width samples and their confidences
        return [(sample.sequence, sample.confidence) for sample in search_space]

    def _fix(self, src: Tensor, top_k: int) -> list[(float, Tensor)]:
        """
        Apply beam search in LSTM.
        :param src: [input_length]
        :param top_k:
        :return:
        """
        # Format input to fit into beam search
        # src = [input_length, 1]
        src = src.unsqueeze(0).T
        return self._beam_search(src, top_k)

    def _fit_one_batch(
        self, batch: Tensor, criterion: nn.Module, optimizer: Optimizer
    ) -> float:
        """
        Fit one batch.
        :param batch: The batch to be fitted.
        :param criterion: Criterion to be used.
        :param optimizer: Optimizer to be used.
        :return: the loss.
        """
        src = batch[0].T
        trg = batch[1].T

        optimizer.zero_grad()

        output = self(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = torch.flatten(trg[1:])  # trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)

        optimizer.step()

        return loss.item()

    def _eval_one_batch(self, batch: Tensor, criterion: nn.Module) -> float:
        """
        Evaluate one batch.
        :param batch: The batch to be evaluated.
        :param criterion: The criterion to be used.
        :return: the loss.
        """
        src = batch[0].T
        trg = batch[1].T

        output = self(src, trg, 0)  # turn off teacher forcing

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = torch.flatten(trg[1:]).view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        return loss.item()

    def _inp_tokenizer(self) -> ModelTokenizer:
        return SequenceTokenizer(
            self.input_length,
            self.src_vocab.sos,
            self.src_vocab.eos,
            self.src_vocab.pad,
        )
