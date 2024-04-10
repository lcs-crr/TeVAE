"""
Lucas Correia
Mercedes-Benz AG
Mercedesstr. 137 | 70327 Stuttgart | Germany

Original paper DOI: 10.3390/s22082886
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers as tfkl


class LWVAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(LWVAE, self).__init__()

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
        rec_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(X, Xhat)

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
        Xhat = self.decoder(Z, training=False)
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
        Xhat = self.decoder(Z, **kwargs)
        return Xhat, Z_mean, Z_logvar, Z


class LWVAE_Encoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features, seed):
        super(LWVAE_Encoder, self).__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.seed = seed
        self.encoder = self.build_encoder()

    def build_encoder(self):
        # Input window
        enc_input = tfkl.Input(shape=(self.seq_len, self.features))
        # LSTM layer
        lstm = tfkl.LSTM(128, return_sequences=False)(enc_input)
        # Transform deterministic BiLSTM output into distribution parameters Z_mean and Z_logvar
        Z_mean = tfkl.Dense(self.latent_dim, name="Z_mean")(lstm)
        Z_logvar = tfkl.Dense(self.latent_dim, name="Z_logvar")(lstm)
        # Create distribution object for reparametrisation trick
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        # Get epsilon for reparametrisation trick
        eps = output_dist.sample(tf.shape(Z_mean), seed=self.seed)
        # Reparametrisation trick
        Z = Z_mean + tf.sqrt(tf.math.exp(Z_logvar)) * eps
        return tf.keras.Model(enc_input, [Z_mean, Z_logvar, Z], name="encoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.encoder(inputs, **kwargs)


class LWVAE_Decoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features, seed):
        super(LWVAE_Decoder, self).__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.seed = seed
        self.decoder = self.build_decoder()

    def build_decoder(self):
        # Latent vector input
        latent_input = tfkl.Input(shape=(self.latent_dim,))
        repeat_latent = tfkl.RepeatVector(self.seq_len)(latent_input)
        # LSTM layer
        lstm = tfkl.LSTM(128, return_sequences=True)(repeat_latent)
        # Map LSTM output to VAE output
        Xhat = tfkl.TimeDistributed(tfkl.Dense(self.features), name="Xhat")(lstm)
        return tf.keras.Model(latent_input, Xhat, name="decoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.decoder(inputs, **kwargs)


if __name__ == "__main__":
    pass
