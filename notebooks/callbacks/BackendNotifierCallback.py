import tensorflow as tf
import keras
import os
import time
import requests


class BackendNotifierCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("🚀 Iniciando entrenamiento pesado. Notificando al sistema central...")
        # Aquí harías un POST a tu API local en Java/Spring Boot
        # requests.post("http://localhost:8080/api/status", json={"estado": "ENTRENANDO"})

    def on_epoch_end(self, epoch, logs=None):
        # Si el error hace una locura, abortamos la misión y avisamos al backend
        if logs.get('loss') is not None and logs.get('loss') > 500:
            print(f"⚠️ ERROR CRÍTICO: El gradiente explotó en la época {epoch}.")
            self.model.stop_training = True
            # requests.post("http://localhost:8080/api/status", json={"estado": "ERROR_MATEMATICO"})

    def on_train_end(self, logs=None):
        print("✅ Entrenamiento finalizado con éxito.")
        # requests.post("http://localhost:8080/api/status", json={"estado": "LISTO"})