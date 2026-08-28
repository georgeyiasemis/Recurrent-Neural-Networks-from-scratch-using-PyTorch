"""Recurrent cell implementations from scratch using PyTorch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.xh = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.hh = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -std, std)

    def forward(
        self,
        input: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)
            hx = (hx, hx)

        hx, cx = hx
        gates = self.xh(input) + self.hh(hx)

        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = cx * f_t + i_t * g_t
        hy = o_t * torch.tanh(cy)

        return hy, cy


class RNNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in {"tanh", "relu"}:
            raise ValueError("Invalid nonlinearity selected for RNN.")

        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -std, std)

    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)

        hy = self.x2h(input) + self.h2h(hx)
        if self.nonlinearity == "tanh":
            hy = torch.tanh(hy)
        else:
            hy = torch.relu(hy)

        return hy


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -std, std)

    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)

        x_t = self.x2h(input)
        h_t = self.h2h(hx)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        return update_gate * hx + (1 - update_gate) * new_gate
