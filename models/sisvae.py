"""
Lucas Correia
Mercedes-Benz AG
Mercedesstr. 137 | 70327 Stuttgart | Germany

Original paper DOI: 10.1109/TNNLS.2020.2980749
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers as tfkl


class SISVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=0.5):
        super(SISVAE, self).__init__()

        # Model
        self.encoder = encoder
        self.decoder = decoder

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.smooth_loss_tracker = tf.keras.metrics.Mean(name="smooth_loss")

        # Modifiable weight for KL-loss
        self.beta = tf.Variable(beta, trainable=False)  # Weight for KL-Loss, can be modified with a callback

    @tf.function
    def loss_fn(self, X, Xhat, Xhat_mean, Xhat_logvar, Z_mean, Z_logvar):
        # Configure distribution with parameters output from decoder
        output_dist = tfp.distributions.MultivariateNormalDiag(loc=Xhat_mean, scale_diag=tf.sqrt(tf.math.exp(Xhat_logvar)))
        # Calculate log probability of sample belongs parametrised distribution
        loglik_loss = output_dist.log_prob(X)

        # Calculate KL Divergence between latent distribution and Gaussian distribution
        kl_loss = tfp.distributions.kl_divergence(
            tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(Z_mean), scale_diag=tf.ones_like(Z_logvar)),
            tfp.distributions.MultivariateNormalDiag(loc=Z_mean, scale_diag=tf.sqrt(tf.math.exp(Z_logvar)))
        )

        # Calculate KL Divergence between current latent distribution and t-1 latent distribution
        smooth_loss = [tfp.distributions.kl_divergence(
            tfp.distributions.MultivariateNormalDiag(loc=Z_mean[:, time_step - 1], scale_diag=tf.sqrt(tf.math.exp(Z_logvar[:, time_step - 1]))),
            tfp.distributions.MultivariateNormalDiag(loc=Z_mean[:, time_step], scale_diag=tf.sqrt(tf.math.exp(Z_logvar[:, time_step]))),
        ) for time_step in range(1, kl_loss.shape[1])]
        smooth_loss = tf.transpose(tf.stack(smooth_loss), perm=[1, 0])

        return -tf.reduce_sum(loglik_loss, axis=1), tf.reduce_sum(kl_loss, axis=1), tf.reduce_sum(smooth_loss, axis=1)

    @tf.function
    def train_step(self, X):
        if isinstance(X, tuple):
            X = X[0]
        with tf.GradientTape() as tape:
            # Forward pass through encoder
            Z_mean, Z_logvar, Z = self.encoder(X, training=True)
            # Forward pass through decoder
            Xhat_mean, Xhat_logvar, Xhat = self.decoder(Z, training=True)
            # Calculate losses from parameters
            negloglik_loss, kl_loss, smooth_loss = self.loss_fn(
                X,
                Xhat,
                Xhat_mean,
                Xhat_logvar,
                Z_mean,
                Z_logvar
            )
            # Calculate total loss from different losses
            total_loss = negloglik_loss + kl_loss + self.beta * smooth_loss
        # Calculate gradients in backward pass
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(negloglik_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.smooth_loss_tracker.update_state(smooth_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.rec_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "smooth_loss": self.smooth_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, X):
        if isinstance(X, tuple):
            X = X[0]
        # Forward pass through encoder
        Z_mean, Z_logvar, Z = self.encoder(X, training=False)
        # Forward pass through decoder
        Xhat_mean, Xhat_logvar, Xhat = self.decoder(Z, training=False)
        # Calculate losses from parameters
        negloglik_loss, kl_loss, smooth_loss = self.loss_fn(
            X,
            Xhat,
            Xhat_mean,
            Xhat_logvar,
            Z_mean,
            Z_logvar
        )
        total_loss = negloglik_loss + kl_loss + self.beta * smooth_loss
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(negloglik_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.smooth_loss_tracker.update_state(smooth_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.rec_loss_tracker,
            self.kl_loss_tracker,
            self.smooth_loss_tracker,
        ]

    @tf.function
    def call(self, inputs, **kwargs):
        Z_mean, Z_logvar, Z = self.encoder(inputs, **kwargs)
        Xhat_mean, Xhat_logvar, Xhat = self.decoder(Z, **kwargs)
        return Xhat_mean, Xhat_logvar, Xhat, Z_mean, Z_logvar, Z


class SISVAE_Encoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features, seed):
        super(SISVAE_Encoder, self).__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.seed = seed
        self.encoder = self.build_encoder()

    def build_encoder(self):
        # Input window
        enc_input = tfkl.Input(shape=(self.seq_len, self.features))
        # GRU layer
        gru = tfkl.GRU(200, return_sequences=True)(enc_input)
        # Transform deterministic BiLSTM output into distribution parameters Z_mean and Z_logvar
        Z_mean = tfkl.TimeDistributed(tfkl.Dense(self.latent_dim, name="Z_mean"))(gru)
        Z_logvar = tfkl.TimeDistributed(tfkl.Dense(self.latent_dim, name="Z_logvar"))(gru)
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


class SISVAE_Decoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features, seed):
        super(SISVAE_Decoder, self).__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.seed = seed
        self.decoder = self.build_decoder()

    def build_decoder(self):
        # Latent vector input
        latent_input = tfkl.Input(shape=(self.seq_len, self.latent_dim,))
        # GRU layer
        gru = tfkl.GRU(200, return_sequences=True)(latent_input)
        # Transform deterministic GRU output into distribution parameters Z_mean and Z_logvar
        Xhat_mean = tfkl.TimeDistributed(tfkl.Dense(self.features), name="Xhat_mean")(gru)
        Xhat_logvar = tfkl.TimeDistributed(tfkl.Dense(self.features), name="Xhat_logvar")(gru)
        # Create distribution object for reparametrisation trick
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        # Get epsilon for reparametrisation trick
        eps = output_dist.sample(tf.shape(Xhat_mean), seed=self.seed)
        # Reparametrisation trick
        Xhat = Xhat_mean + tf.sqrt(tf.math.exp(Xhat_logvar)) * eps
        return tf.keras.Model(latent_input, [Xhat_mean, Xhat_logvar, Xhat], name="decoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.decoder(inputs, **kwargs)


if __name__ == "__main__":
    pass
