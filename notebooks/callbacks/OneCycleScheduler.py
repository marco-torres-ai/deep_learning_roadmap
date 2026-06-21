import math
import tensorflow as tf
import keras

class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        # Si no le pasas un inicio, asume que es el 10% del máximo
        self.start_rate = start_rate or max_rate / 10
        # El último 10% del entrenamiento será para el "aterrizaje"
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        # La tasa final será súper diminuta
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0

    # Función matemática para calcular en qué punto de la "montaña" estamos
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)

    def on_batch_begin(self, batch, logs=None):
        if self.iteration < self.half_iteration:
            # Fase 1: Subida
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            # Fase 2: Bajada
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            # Fase 3: Aterrizaje final milimétrico
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
        self.iteration += 1
        # Inyectamos la nueva tasa directamente en el optimizador de tu modelo
        learning_rate = self.model.optimizer.learning_rate
        if hasattr(learning_rate, "assign"):
            learning_rate.assign(rate)
        else:
            self.model.optimizer.learning_rate = rate