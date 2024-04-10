"""
Lucas Correia
Mercedes-Benz AG
Mercedesstr. 137 | 70327 Stuttgart | Germany

Original paper DOI: 10.1016/j.engappai.2021.104354
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers as tfkl


class VASP(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(VASP, self).__init__()

        # Model
        self.encoder = encoder
        self.decoder = decoder

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @tf.function
    def loss_fn(self, X, Xhat, Z_mean, Z_logvar):
        # Calculate reconstruction error
        rec_loss = (X-Xhat)**2
        rec_loss = tf.reduce_sum(rec_loss, axis=2)

        # Calculate KL Divergence between latent distribution and Gaussian distribution
        kl_loss = tfp.distributions.kl_divergence(
            tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(Z_mean), scale_diag=tf.ones_like(Z_logvar)),
            tfp.distributions.MultivariateNormalDiag(loc=Z_mean, scale_diag=tf.sqrt(tf.math.exp(Z_logvar)))
        )

        return tf.reduce_sum(rec_loss, axis=1), kl_loss

    @tf.function
    def train_step(self, X):
        if isinstance(X, tuple):
            X = X[0]
        with tf.GradientTape() as tape:
            # Forward pass through encoder
            Z_mean, Z_logvar, Z = self.encoder(X, training=True)
            # Forward pass through decoder
            Xhat = self.decoder(Z, training=True)
            # Calculate losses from parameters
            rec_loss, kl_loss = self.loss_fn(
                X,
                Xhat,
                Z_mean,
                Z_logvar
            )
            # Calculate total loss from different losses
            total_loss = rec_loss + kl_loss
        # Calculate gradients in backward pass
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.rec_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, X):
        if isinstance(X, tuple):
            X = X[0]
        # Forward pass through encoder
        Z_mean, Z_logvar, Z = self.encoder(X, training=False)
        # Forward pass through decoder
        Xhat = self.decoder(Z_mean, training=False)
        # Calculate losses from parameters
        rec_loss, kl_loss = self.loss_fn(
            X,
            Xhat,
            Z_mean,
            Z_logvar
        )
        total_loss = rec_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.rec_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def call(self, inputs, **kwargs):
        Z_mean, Z_logvar, Z = self.encoder(inputs, **kwargs)
        Xhat = self.decoder(Z_mean, **kwargs)
        return Xhat, Z_mean, Z_logvar, Z


class VASP_Encoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features, seed):
        super(VASP_Encoder, self).__init__()

        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.seed = seed
        self.encoder = self.build_encoder()

    def build_encoder(self):
        # Input window
        enc_input = tfkl.Input(shape=(self.seq_len, self.features))
        #  FC layer
        fc = tfkl.TimeDistributed((tfkl.Dense(80)))(enc_input)
        #  LSTM layer
        lstm = tfkl.LSTM(65, return_sequences=False, dropout=0.2, recurrent_dropout=0.1)(fc)
        # FC Layer
        fc = (tfkl.Dense(13))(lstm)
        # Transform deterministic BiLSTM output into distribution parameters Z_mean and Z_logvar
        Z_mean = tfkl.Dense(self.latent_dim, name="Z_mean")(fc)
        Z_logvar = tfkl.Dense(self.latent_dim, name="Z_logvar")(fc)
        # Create distribution object for reparametrisation trick
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        # Get epsilon for reparametrisation trick
        eps = output_dist.sample(tf.shape(Z_mean))
        # Reparametrisation trick
        Z = Z_mean + tf.sqrt(tf.math.exp(Z_logvar)) * eps
        return tf.keras.Model(enc_input, [Z_mean, Z_logvar, Z], name="encoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.encoder(inputs, **kwargs)


class VASP_Decoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features, seed):
        super(VASP_Decoder, self).__init__()
        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.seed = seed
        self.decoder = self.build_decoder()

    def build_decoder(self):
        # Latent vector input
        latent_input = tfkl.Input(shape=(self.latent_dim,))
        # FC Layer
        fc = (tfkl.Dense(13))(latent_input)
        # FC Layer
        fc = (tfkl.Dense(65))(fc)
        # Repeat vector as many times as the window length
        rep = tfkl.RepeatVector(self.seq_len)(fc)
        # LSTM layer
        lstm = tfkl.LSTM(65, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(rep)
        #  FC layer
        fc = tfkl.TimeDistributed((tfkl.Dense(80)))(lstm)
        # FC Layer
        Xhat = tfkl.TimeDistributed(tfkl.Dense(self.features), name="Xhat_mean")(fc)
        return tf.keras.Model(latent_input, Xhat, name="decoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.decoder(inputs, **kwargs)


if __name__ == "__main__":
    pass
