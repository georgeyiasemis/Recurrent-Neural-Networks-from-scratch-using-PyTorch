# Recurrent Neural Networks from Scratch using PyTorch

Educational implementations of recurrent cells and multi-layer models in pure PyTorch — no `nn.RNN`, `nn.LSTM`, or `nn.GRU`.

## What's included

**Cells** (`rnncells.py`):

- `RNNCell` — vanilla RNN with tanh or ReLU
- `LSTMCell` — long short-term memory cell
- `GRUCell` — gated recurrent unit

**Models** (`rnnmodels.py`):

- `SimpleRNN` — stacked RNN
- `LSTM` — stacked LSTM
- `GRU` — stacked GRU
- `BidirRecurrentModel` — bidirectional RNN / LSTM / GRU

All models accept input of shape `(batch_size, sequence_length, input_size)` and return `(batch_size, output_size)` using the final time step.

## LSTM cell architecture

![LSTM block architecture](https://user-images.githubusercontent.com/71031687/112729023-5efd7780-8f2a-11eb-88a7-32c3861b91a5.jpg)

## Requirements

- Python 3.10+
- PyTorch 2.0+

Install dependencies:

```bash
pip install -e .
```

Or install PyTorch directly:

```bash
pip install torch
```

## Quick start

```python
import torch
from rnnmodels import LSTM

model = LSTM(
    input_size=16,
    hidden_size=32,
    num_layers=2,
    bias=True,
    output_size=4,
)

x = torch.randn(8, 12, 16)  # batch, seq_len, features
logits = model(x)
print(logits.shape)  # torch.Size([8, 4])
```

Run the included training demo:

```bash
python examples/demo.py
```

Models follow the input tensor's device automatically — pass CUDA or MPS tensors without extra setup:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
logits = model(x)
```

## Project layout

```
rnncells.py      # RNN, LSTM, GRU cells
rnnmodels.py     # Multi-layer and bidirectional models
examples/demo.py # Short training loop over all model types
pyproject.toml   # Package metadata and dependencies
```

## License

MIT — see [LICENSE](LICENSE).
