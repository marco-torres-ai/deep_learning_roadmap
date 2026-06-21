import keras
import requests
import tensorflow as tf
import keras

class AlertaBackendCallback(keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        # Este código se ejecuta exactamente cuando el entrenamiento finaliza (o se detiene)
        print("¡Entrenamiento terminado! Notificando al backend...")
        
        # Ejemplo: Enviar un POST a tu API local
        payload = {"mensaje": "El modelo Wide & Deep ya está listo.", "loss_final": logs.get('loss')}
        # requests.post("http://localhost:8080/api/notificaciones", json=payload)
        
    def on_epoch_end(self, epoch, logs=None):
        # Puedes vigilar si hay un error matemático crítico (NaN) y detenerlo tú mismo
        if logs.get('loss') > 100:
            print(f"¡Alerta! El error explotó en la época {epoch}. Deteniendo...")
            self.model.stop_training = True
