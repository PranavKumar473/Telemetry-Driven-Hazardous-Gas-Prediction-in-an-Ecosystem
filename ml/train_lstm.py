from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from ml.features import TARGET_GASES, HORIZONS, get_feature_columns

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
SEQ_LEN = 12
TARGET_COLS = [f"{g}_target_{h}h" for g in TARGET_GASES for h in HORIZONS]


class GasLSTM(nn.Module):
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
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
    xs, ys = [], []
    for i in range(seq_len - 1, len(X)):
        xs.append(X[i - seq_len + 1 : i + 1])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PyTorch LSTM on device: {device}")

    train_df = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_features.csv")
    feature_cols = get_feature_columns(train_df)

    # Fit scalers strictly on training data to prevent leakage
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(train_df[feature_cols].values)
    y_train_s = y_scaler.fit_transform(train_df[TARGET_COLS].values)
    X_val_s = x_scaler.transform(val_df[feature_cols].values)
    y_val_s = y_scaler.transform(val_df[TARGET_COLS].values)

    X_train_seq, y_train_seq = make_sequences(X_train_s, y_train_s)
    X_val_seq, y_val_seq = make_sequences(X_val_s, y_val_s)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq)),
        batch_size=64,
        shuffle=True,
    )
    X_val_t = torch.from_numpy(X_val_seq).to(device)
    y_val_t = torch.from_numpy(y_val_seq).to(device)

    model = GasLSTM(input_size=len(feature_cols), output_size=len(TARGET_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(1, 26):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_size": len(feature_cols),
                    "output_size": len(TARGET_COLS),
                    "seq_len": SEQ_LEN,
                },
                MODELS_DIR / "lstm_model.pt",
            )

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/25 | Val MSE (scaled): {val_loss:.4f}")

    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, MODELS_DIR / "lstm_scalers.joblib")
    print(f"Saved best LSTM checkpoint (Val MSE: {best_val_loss:.4f}) and scalers to models\\")


if __name__ == "__main__":
    main()