"""
Lucas Correia
Mercedes-Benz AG
Mercedesstr. 137 | 70327 Stuttgart | Germany

Original paper DOI: 10.1145/3292500.3330672
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers as tfkl


class OmniAnomaly(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(OmniAnomaly, self).__init__()

        # Model
        self.encoder = encoder
        self.decoder = decoder

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.lgssm_loss_tracker = tf.keras.metrics.Mean(name="lgssm_loss")

    @tf.function
    def loss_fn(self, X, Xhat, Xhat_mean, Xhat_logvar, Z, Z_mean, Z_logvar):
        # Configure distribution with parameters output from decoder
        output_dist = tfp.distributions.MultivariateNormalDiag(loc=Xhat_mean, scale_diag=tf.sqrt(tf.math.exp(Xhat_logvar)))
        # Calculate log probability of sample belongs parametrised distribution
        loglik_loss = output_dist.log_prob(X)

        # Calculate KL Divergence between latent distribution and Gaussian distribution
        kl_loss = tfp.distributions.kl_divergence(
            tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(Z_mean), scale_diag=tf.ones_like(Z_logvar)),
            tfp.distributions.MultivariateNormalDiag(loc=Z_mean, scale_diag=tf.sqrt(tf.math.exp(Z_logvar)))
        )

        lgssm_dist = tfp.distributions.LinearGaussianStateSpaceModel(
            num_timesteps=Z.shape[1],
            transition_matrix=tf.linalg.LinearOperatorIdentity(Z.shape[-1]),
            transition_noise=tfp.distributions.MultivariateNormalDiag(scale_diag=tf.ones([Z.shape[-1]])),
            observation_matrix=tf.linalg.LinearOperatorIdentity(Z.shape[-1]),
            observation_noise=tfp.distributions.MultivariateNormalDiag(scale_diag=tf.ones([Z.shape[-1]])),
            initial_state_prior=tfp.distributions.MultivariateNormalDiag(scale_diag=tf.ones([Z.shape[-1]]))
        )
        # Calculate log probability of sample belongs parametrised distribution
        lgssm_loss = lgssm_dist.log_prob(Z)

        return -tf.reduce_sum(loglik_loss, axis=1), tf.reduce_sum(kl_loss, axis=1), -lgssm_loss

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
            negloglik_loss, kl_loss, lgssm_loss = self.loss_fn(
                X,
                Xhat,
                Xhat_mean,
                Xhat_logvar,
                Z,
                Z_mean,
                Z_logvar
            )
            # Calculate total loss from different losses
            total_loss = negloglik_loss + kl_loss + lgssm_loss
        # Calculate gradients in backward pass
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(negloglik_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.lgssm_loss_tracker.update_state(lgssm_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.rec_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "lgssm_loss": self.lgssm_loss_tracker.result(),
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
        negloglik_loss, kl_loss, lgssm_loss = self.loss_fn(
            X,
            Xhat,
            Xhat_mean,
            Xhat_logvar,
            Z,
            Z_mean,
            Z_logvar
        )
        total_loss = negloglik_loss + kl_loss + lgssm_loss
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(negloglik_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.lgssm_loss_tracker.update_state(lgssm_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.rec_loss_tracker,
            self.kl_loss_tracker,
            self.lgssm_loss_tracker,
        ]

    @tf.function
    def call(self, inputs, **kwargs):
        # Forward pass through encoder
        Z_mean, Z_logvar, Z = self.encoder(inputs, training=False)
        # Forward pass through decoder
        Xhat_mean, Xhat_logvar, Xhat = self.decoder(Z, training=False)
        return Xhat_mean, Xhat_logvar, Xhat, Z_mean, Z_logvar, Z


class OmniAnomaly_Encoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features, seed):
        super(OmniAnomaly_Encoder, self).__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.seed = seed
        self.encoder = self.build_encoder()

    def build_encoder(self):
        enc_input = tfkl.Input(shape=(self.seq_len, self.features))
        x = tfkl.GRU(500, return_sequences=True, activity_regularizer=tf.keras.regularizers.L2(l2=1e-4))(enc_input)
        h = tfkl.TimeDistributed(tfkl.Dense(500, activation="relu", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(x)
        Z_mean = tfkl.TimeDistributed(tfkl.Dense(self.latent_dim, name="Z_mean", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(h)
        Z_logvar = tfkl.TimeDistributed(tfkl.Dense(self.latent_dim, name="Z_logvar", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(h)
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        eps = output_dist.sample(tf.shape(Z_mean), seed=self.seed)
        Z = Z_mean + tf.sqrt(tf.math.exp(Z_logvar)) * eps + 1e-4
        K = 20
        # planar normalizing flow
        for k in range(K):
            Z = Z + tfkl.Dense(self.latent_dim, use_bias=False)(tfkl.Dense(self.latent_dim, activation='tanh')(Z))
        return tf.keras.Model(enc_input, [Z_mean, Z_logvar, Z], name="encoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.encoder(inputs, **kwargs)


class OmniAnomaly_Decoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features, seed):
        super(OmniAnomaly_Decoder, self).__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.seed = seed
        self.decoder = self.build_decoder()

    def build_decoder(self):
        dec_input = tfkl.Input(shape=(self.seq_len, self.latent_dim))
        x = tfkl.GRU(500, return_sequences=True, activity_regularizer=tf.keras.regularizers.L2(l2=1e-4))(dec_input)
        h = tfkl.TimeDistributed(tfkl.Dense(500, activation="relu", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(x)
        Xhat_mean = tfkl.TimeDistributed(tfkl.Dense(self.features, name="Xhat_mean", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(h)
        Xhat_logvar = tfkl.TimeDistributed(tfkl.Dense(self.features, name="Xhat_logvar", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(h)
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        eps = output_dist.sample(tf.shape(Xhat_mean), seed=self.seed)
        Xhat = Xhat_mean + tf.sqrt(tf.math.exp(Xhat_logvar)) * eps
        return tf.keras.Model(dec_input, [Xhat_mean, Xhat_logvar, Xhat], name="decoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.decoder(inputs, **kwargs)


if __name__ == "__main__":
    pass
