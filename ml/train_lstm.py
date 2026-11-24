"""Train the stacked LSTM on 24h sliding windows (16 features -> 9 targets) with masked Huber loss."""
from ml.config import SEQ_LEN  # noqa: F401  (re-exported for backend imports)
from ml.models import GasLSTM
from ml.nn_training import load_windows, train_model
from ml.windowing import TARGET_COLS  # noqa: F401  (re-exported for backend imports)

HPARAMS = {"hidden_size": 64, "num_layers": 2}


def main() -> None:
    data = load_windows()
    model = GasLSTM(input_size=len(data["feature_names"]), output_size=len(data["target_names"]), **HPARAMS)
    train_model(model, name="lstm", model_type="lstm", hparams=HPARAMS, data=data)


if __name__ == "__main__":
    main()
