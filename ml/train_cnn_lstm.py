"""Train the 1D-CNN + LSTM hybrid on the same 24h windows and masked Huber loss as the LSTM."""
from ml.models import CNNLSTM
from ml.nn_training import load_windows, train_model

HPARAMS = {"conv_channels": 32, "hidden_size": 64}


def main() -> None:
    data = load_windows()
    model = CNNLSTM(input_size=len(data["feature_names"]), output_size=len(data["target_names"]), **HPARAMS)
    train_model(model, name="cnn_lstm", model_type="cnn_lstm", hparams=HPARAMS, data=data)


if __name__ == "__main__":
    main()
