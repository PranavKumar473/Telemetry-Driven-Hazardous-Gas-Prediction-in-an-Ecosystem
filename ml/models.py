"""Neural network architectures for multi-horizon gas forecasting.

All models take a window of shape (batch, seq_len, n_features) and return
(batch, n_targets) scaled predictions for 3 gases x 3 horizons.
"""
import torch
import torch.nn as nn


class GasLSTM(nn.Module):
    """Stacked LSTM (Unit IV): reads the 24h window step by step, predicts from the last hidden state."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 9):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)               # (B, T, H)
        return self.head(out[:, -1, :])     # (B, output_size)


class CNNLSTM(nn.Module):
    """
    1D-CNN + LSTM hybrid (Units II-IV).
    Conv1d layers learn short local patterns (spikes, ramps) across a few hours,
    MaxPool halves the sequence, then an LSTM models the longer-range trend.

    (B, T, F) -> (B, F, T) -> conv/pool -> (B, C, T/2) -> (B, T/2, C) -> LSTM -> (B, output_size)
    """

    def __init__(self, input_size: int, conv_channels: int = 32, hidden_size: int = 64, output_size: int = 9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.permute(0, 2, 1))   # Conv1d expects (B, channels=F, length=T)
        out, _ = self.lstm(z.permute(0, 2, 1))
        return self.head(out[:, -1, :])


MODEL_REGISTRY = {"lstm": GasLSTM, "cnn_lstm": CNNLSTM}


def build_model(model_type: str, input_size: int, output_size: int, **hparams) -> nn.Module:
    return MODEL_REGISTRY[model_type](input_size=input_size, output_size=output_size, **hparams)


def load_checkpoint(path) -> tuple[nn.Module, dict]:
    """Rebuild a model from a checkpoint saved by ml.nn_training.train_model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("model_type", "lstm"), ckpt["input_size"], ckpt["output_size"], **ckpt.get("hparams", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
