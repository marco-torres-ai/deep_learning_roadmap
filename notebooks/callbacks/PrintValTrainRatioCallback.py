
import keras


class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        train_loss = logs.get("loss")
        if val_loss is not None and train_loss is not None:
            ratio = val_loss / train_loss
            print(f"Epoch {epoch + 1}: Validation/Training Loss Ratio: {ratio:.4f}")    