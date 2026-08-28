"""Train a small sequence classifier with the from-scratch RNN models."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnnmodels import BidirRecurrentModel, GRU, LSTM, SimpleRNN


def main() -> None:
    torch.manual_seed(42)

    batch_size = 8
    seq_len = 12
    input_size = 16
    hidden_size = 32
    num_layers = 2
    output_size = 4

    x = torch.randn(batch_size, seq_len, input_size)
    y = torch.randint(0, output_size, (batch_size,))

    models: dict[str, nn.Module] = {
        "RNN": SimpleRNN(input_size, hidden_size, num_layers, True, output_size),
        "LSTM": LSTM(input_size, hidden_size, num_layers, True, output_size),
        "GRU": GRU(input_size, hidden_size, num_layers, True, output_size),
        "BiLSTM": BidirRecurrentModel(
            "LSTM", input_size, hidden_size, num_layers, True, output_size
        ),
    }

    criterion = nn.CrossEntropyLoss()

    for name, model in models.items():
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        for _ in range(5):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        print(f"{name}: loss={loss.item():.4f}, output shape={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
