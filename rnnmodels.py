"""Multi-layer and bidirectional RNN models built from scratch."""

from __future__ import annotations

import torch
import torch.nn as nn

from rnncells import GRUCell, LSTMCell, RNNCell


def _make_cell(mode: str, input_size: int, hidden_size: int, bias: bool) -> nn.Module:
    if mode == "LSTM":
        return LSTMCell(input_size, hidden_size, bias)
    if mode == "GRU":
        return GRUCell(input_size, hidden_size, bias)
    if mode == "RNN_TANH":
        return RNNCell(input_size, hidden_size, bias, "tanh")
    if mode == "RNN_RELU":
        return RNNCell(input_size, hidden_size, bias, "relu")
    raise ValueError("Invalid RNN mode selected.")


def _init_hidden(
    num_layers: int,
    batch_size: int,
    hidden_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.zeros(num_layers, batch_size, hidden_size, device=device, dtype=dtype)


class SimpleRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool,
        output_size: int,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        if activation not in {"tanh", "relu"}:
            raise ValueError("Invalid activation.")

        self.rnn_cell_list = nn.ModuleList(
            [
                RNNCell(
                    input_size if layer == 0 else hidden_size,
                    hidden_size,
                    bias,
                    activation,
                )
                for layer in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hx is None:
            h0 = _init_hidden(
                self.num_layers,
                input.size(0),
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )
        else:
            h0 = hx

        hidden = [h0[layer] for layer in range(self.num_layers)]
        outs: list[torch.Tensor] = []

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                cell_input = input[:, t, :] if layer == 0 else hidden[layer - 1]
                hidden[layer] = self.rnn_cell_list[layer](cell_input, hidden[layer])
            outs.append(hidden[-1])

        out = self.fc(outs[-1].squeeze())
        return out


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool,
        output_size: int,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList(
            [
                LSTMCell(
                    input_size if layer == 0 else hidden_size,
                    hidden_size,
                    bias,
                )
                for layer in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hx is None:
            h0 = _init_hidden(
                self.num_layers,
                input.size(0),
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )
        else:
            h0 = hx

        hidden = [(h0[layer], h0[layer]) for layer in range(self.num_layers)]
        outs: list[torch.Tensor] = []

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                cell_input = input[:, t, :] if layer == 0 else hidden[layer - 1][0]
                hidden[layer] = self.rnn_cell_list[layer](cell_input, hidden[layer])
            outs.append(hidden[-1][0])

        out = self.fc(outs[-1].squeeze())
        return out


class GRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool,
        output_size: int,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList(
            [
                GRUCell(
                    input_size if layer == 0 else hidden_size,
                    hidden_size,
                    bias,
                )
                for layer in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hx is None:
            h0 = _init_hidden(
                self.num_layers,
                input.size(0),
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )
        else:
            h0 = hx

        hidden = [h0[layer] for layer in range(self.num_layers)]
        outs: list[torch.Tensor] = []

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                cell_input = input[:, t, :] if layer == 0 else hidden[layer - 1]
                hidden[layer] = self.rnn_cell_list[layer](cell_input, hidden[layer])
            outs.append(hidden[-1])

        out = self.fc(outs[-1].squeeze())
        return out


class BidirRecurrentModel(nn.Module):
    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool,
        output_size: int,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        if mode not in {"LSTM", "GRU", "RNN_TANH", "RNN_RELU"}:
            raise ValueError("Invalid RNN mode selected.")

        self.rnn_cell_list = nn.ModuleList(
            [
                _make_cell(
                    mode,
                    input_size if layer == 0 else hidden_size,
                    hidden_size,
                    bias,
                )
                for layer in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input: torch.Tensor, hx: torch.Tensor | None = None) -> torch.Tensor:
        h0 = _init_hidden(
            self.num_layers,
            input.size(0),
            self.hidden_size,
            device=input.device,
            dtype=input.dtype,
        )

        if self.mode == "LSTM":
            hidden_forward = [(h0[layer], h0[layer]) for layer in range(self.num_layers)]
            hidden_backward = [(h0[layer], h0[layer]) for layer in range(self.num_layers)]
        else:
            hidden_forward = [h0[layer] for layer in range(self.num_layers)]
            hidden_backward = [h0[layer] for layer in range(self.num_layers)]

        outs: list[torch.Tensor] = []
        outs_rev: list[torch.Tensor] = []

        for t in range(input.shape[1]):
            for layer in range(self.num_layers):
                if self.mode == "LSTM":
                    if layer == 0:
                        h_forward_l = self.rnn_cell_list[layer](
                            input[:, t, :],
                            hidden_forward[layer],
                        )
                        h_back_l = self.rnn_cell_list[layer](
                            input[:, -(t + 1), :],
                            hidden_backward[layer],
                        )
                    else:
                        h_forward_l = self.rnn_cell_list[layer](
                            hidden_forward[layer - 1][0],
                            hidden_forward[layer],
                        )
                        h_back_l = self.rnn_cell_list[layer](
                            hidden_backward[layer - 1][0],
                            hidden_backward[layer],
                        )
                elif layer == 0:
                    h_forward_l = self.rnn_cell_list[layer](
                        input[:, t, :],
                        hidden_forward[layer],
                    )
                    h_back_l = self.rnn_cell_list[layer](
                        input[:, -(t + 1), :],
                        hidden_backward[layer],
                    )
                else:
                    h_forward_l = self.rnn_cell_list[layer](
                        hidden_forward[layer - 1],
                        hidden_forward[layer],
                    )
                    h_back_l = self.rnn_cell_list[layer](
                        hidden_backward[layer - 1],
                        hidden_backward[layer],
                    )

                hidden_forward[layer] = h_forward_l
                hidden_backward[layer] = h_back_l

            if self.mode == "LSTM":
                outs.append(h_forward_l[0])
                outs_rev.append(h_back_l[0])
            else:
                outs.append(h_forward_l)
                outs_rev.append(h_back_l)

        out = torch.cat((outs[-1].squeeze(), outs_rev[0].squeeze()), dim=1)
        return self.fc(out)
