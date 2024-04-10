"""
Lucas Correia
Mercedes-Benz AG
Mercedesstr. 137 | 70327 Stuttgart | Germany
"""

import numpy as np
import tensorflow as tf


class KL_annealing(tf.keras.callbacks.Callback):
    def __init__(self, annealing_epochs=25, annealing_type="normal", grace_period=25, start=1e-3, end=1e0):
        super(KL_annealing, self).__init__()
        self.annealing_epochs = annealing_epochs
        self.annealing_type = annealing_type
        self.grace_period = grace_period
        self.grace_period_idx = np.maximum(0, grace_period - 1)  # Starting from 0
        self.start = start
        self.end = end
        if annealing_type in ["cyclical", "monotonic"]:
            self.beta_values = np.linspace(start, end, annealing_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        shifted_epochs = tf.math.maximum(0.0, epoch - self.grace_period_idx)
        if epoch < self.grace_period_idx or self.annealing_type == "normal":
            step_size = (self.start / self.grace_period)
            new_value = step_size * (epoch % self.grace_period)
            self.model.beta.assign(new_value)
        elif self.annealing_type == "monotonic":
            new_value = self.beta_values[min(epoch, self.annealing_epochs - 1)]
            self.model.beta.assign(new_value)
        elif self.annealing_type == "cyclical":
            new_value = self.beta_values[int(shifted_epochs % self.annealing_epochs)]
            self.model.beta.assign(new_value)
        print(f"Beta value: {self.model.beta.numpy():.10f}, cycle epoch {int(shifted_epochs % self.annealing_epochs)}")


if __name__ == "__main__":
    pass
