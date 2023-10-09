import random
from pathlib import Path

import torch
from bidict import bidict
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.types import Device

from src.styler2_0.models.model_base import ModelBase
from styler2_0.preprocessing.model_preprocessing import load_vocabs

# from styler2_0.utils.checkstyle import run_checkstyle_on_str
# from styler2_0.utils.tokenize import (
#     CheckstyleToken,
#     ProcessedSourceFile,
#     tokenize_java_code,
# )
from styler2_0.utils.utils import load_yaml_file


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
    CONFIG_FILE = Path("lstm.yaml")

    @classmethod
    def build_from_config(cls) -> "LSTM":
        """
        Build the model from the configuration file.
        :return: the model.
        """
        lstm_params = load_yaml_file(cls.CONFIGS_PATH / cls.CONFIG_FILE)
        input_length = lstm_params["input_length"]
        output_length = lstm_params["output_length"]
        src_vocab, trg_vocab = load_vocabs(
            lstm_params["src_vocab"], lstm_params["trg_vocab"]
        )
        encoder = LSTMEncoder(
            input_length,
            lstm_params["enc_emb_dim"],
            lstm_params["hidden_dim"],
            lstm_params["n_layers"],
            lstm_params["enc_dropout"],
        )
        decoder = LSTMDecoder(
            len(trg_vocab),
            lstm_params["dec_emb_dim"],
            lstm_params["hidden_dim"],
            lstm_params["n_layers"],
            lstm_params["dec_dropout"],
        )
        return LSTM(
            encoder,
            decoder,
            lstm_params["device"],
            input_length,
            output_length,
            src_vocab,
            trg_vocab,
        )

    def __init__(
        self,
        encoder: LSTMEncoder,
        decoder: LSTMDecoder,
        device: Device,
        input_length: int,
        output_length: int,
        src_vocab: bidict[int, str],
        trg_vocab: bidict[int, str],
    ) -> None:
        """
        Initialize the model.
        :param encoder: the encoder
        :param decoder: the decoder
        :param device: the device
        :param input_length: input length
        :param output_length: output length
        """
        super().__init__(input_length, output_length, src_vocab, trg_vocab)
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

    # def predict(self, code: str, config: Path, version: str) -> str:
    #     """
    #     Predict the translation of a code.
    #     :param version:
    #     :param config:
    #     :param code:
    #     :return:
    #     """
    #     report = run_checkstyle_on_str(code, version, config)
    #     assert len(report.violations) == 1, "Code contains too many violations!"
    #
    #     tokenized_code = tokenize_java_code(code)
    #     processed_code = ProcessedSourceFile(None, tokenized_code, report)
    #
    #     inp_tokens = []
    #     pick = False
    #     for token in processed_code.tokens:
    #         if isinstance(token, CheckstyleToken) and token.is_starting:
    #             pick = True
    #         elif isinstance(token, CheckstyleToken) and not token.is_starting:
    #             pick = False
    #         if pick:
    #             inp_tokens.append(token)
    #
    #     inp_tokens = (
    #         [self.src_vocab.inverse["<SOS>"]]
    #         + list(map(lambda t: self.src_vocab.inverse[str(t)], inp_tokens))
    #         + [self.src_vocab.inverse["<EOS>"]]
    #     )
    #     inp_tokens.extend(
    #         [self.src_vocab.inverse["<PAD>"]] * (self.input_length - len(inp_tokens))
    #     )
    #
    #     src = torch.tensor(inp_tokens).to(self.device)
    #     trg = torch.zeros(self.output_length, 1, len(self.trg_vocab)).to(self.device)
    #     with torch.no_grad():
    #         output = self(src.T, trg.T, 0)  # turn off teacher forcing
    #     output = output.argmax(2)
    #     return " ".join(map(lambda i: self.trg_vocab[i], output))
