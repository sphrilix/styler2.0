import random
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.types import Device

from src.styler2_0.models.model_base import ModelBase
from styler2_0.utils.utils import load_yaml_file


class LSTMEncoder(nn.Module):
    def __init__(
        self, input_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float
    ) -> None:
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(
            emb_dim, hid_dim, n_layers, bidirectional=True, dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> (Tensor, Tensor):
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
    CONFIG_FILE = Path("lstm.yaml")

    @classmethod
    def build_from_config(cls) -> "LSTM":
        lstm_params = load_yaml_file(cls.CONFIGS_PATH / cls.CONFIG_FILE)
        input_length = lstm_params["input_length"]
        output_length = lstm_params["output_length"]
        encoder = LSTMEncoder(
            input_length,
            lstm_params["enc_emb_dim"],
            lstm_params["hidden_dim"],
            lstm_params["n_layers"],
            lstm_params["enc_dropout"],
        )
        decoder = LSTMDecoder(
            output_length,
            lstm_params["dec_emb_dim"],
            lstm_params["hidden_dim"],
            lstm_params["n_layers"],
            lstm_params["dec_dropout"],
        )
        return LSTM(
            encoder, decoder, lstm_params["device"], input_length, output_length
        )

    def __init__(
        self,
        encoder: LSTMEncoder,
        decoder: LSTMDecoder,
        device: Device,
        input_length: int,
        output_length: int,
    ) -> None:
        super().__init__(input_length, output_length)
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(
        self, src: Tensor, trg: Tensor, teacher_forcing_ratio: float = 0.5
    ) -> Tensor:
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

    def _fit_one_batch(
        self, batch: Tensor, criterion: nn.Module, optimizer: Optimizer
    ) -> float:
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
        src = batch[0].T
        trg = batch[1].T

        output = self(src, trg, 0)  # turn off teacher forcing

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = torch.flatten(trg[1:])  # .view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        return loss.item()
