"""
Lucas Correia
Mercedes-Benz AG
Mercedesstr. 137 | 70327 Stuttgart | Germany

Original paper DOI: 10.1109/ICMLA.2018.00207
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers as tfkl


class VSVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, vs, beta=1e-8, att_beta=1e0):
        super(VSVAE, self).__init__()

        # Model
        self.encoder = encoder
        self.decoder = decoder
        self.vs = vs

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.attkl_loss_tracker = tf.keras.metrics.Mean(name="attkl_loss")

        # Modifiable weight for KL-loss
        self.beta = tf.Variable(beta, trainable=False)  # Weight for KL-Loss
        self.att_beta = tf.constant(att_beta)  # Is not modified

    @tf.function
    def loss_fn(self, X, Xhat, Xhat_mean, Xhat_logvar, Z_mean, Z_logvar, A_mean, A_logvar):
        # Configure distribution with parameters output from decoder
        output_dist = tfp.distributions.MultivariateNormalDiag(loc=Xhat_mean, scale_diag=tf.sqrt(tf.math.exp(Xhat_logvar)))
        # Calculate log probability of sample belongs parametrised distribution
        loglik_loss = output_dist.log_prob(X)

        # Calculate KL Divergence between latent distribution and Gaussian distribution
        kl_loss = tfp.distributions.kl_divergence(
            tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(Z_mean), scale_diag=tf.ones_like(Z_logvar)),
            tfp.distributions.MultivariateNormalDiag(loc=Z_mean, scale_diag=tf.sqrt(tf.math.exp(Z_logvar)))
        )

        # Calculate KL divergence between context vector distribution and Gaussian distribution
        attkl_loss = tfp.distributions.kl_divergence(
            tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(A_mean), scale_diag=tf.ones_like(A_logvar)),
            tfp.distributions.MultivariateNormalDiag(loc=A_mean, scale_diag=tf.sqrt(tf.math.exp(A_logvar)))
            )

        return -tf.reduce_sum(loglik_loss, axis=1), kl_loss, tf.reduce_sum(attkl_loss, axis=1)

    @tf.function
    def train_step(self, X):
        if isinstance(X, tuple):
            X = X[0]
        with tf.GradientTape() as tape:
            # Encoder is fed with input window
            Z_mean, Z_logvar, Z, states = self.encoder(X, training=True)
            # VS is fed with the hidden states from the encoder
            A_mean, A_logvar, A = self.vs(states, training=True)
            # Decoder is fed with the latent vector from the encoder and the attention matrix from VMS mechanism
            Xhat_mean, Xhat_logvar, Xhat = self.decoder([Z, A], training=True)
            # Calculate losses from parameters
            negloglik_loss, kl_loss, attkl_loss = self.loss_fn(
                X,
                Xhat,
                Xhat_mean,
                Xhat_logvar,
                Z_mean,
                Z_logvar,
                A_mean,
                A_logvar
            )
            # Calculate total loss from different losses
            total_loss = negloglik_loss + self.beta * (kl_loss + self.att_beta * attkl_loss)
        # Calculate gradients in backward pass
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(negloglik_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.attkl_loss_tracker.update_state(attkl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.rec_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "attkl_loss": self.attkl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, X):
        if isinstance(X, tuple):
            X = X[0]
        # Encoder is fed with input window
        Z_mean, Z_logvar, Z, states = self.encoder(X, training=False)
        # VS is fed with the hidden states from the encoder
        A_mean, A_logvar, A = self.vs(states, training=False)
        # Decoder is fed with the latent vector from the encoder and the attention matrix from VMS mechanism
        Xhat_mean, Xhat_logvar, Xhat = self.decoder([Z, A], training=False)
        # Calculate losses from parameters
        negloglik_loss, kl_loss, attkl_loss = self.loss_fn(
            X,
            Xhat,
            Xhat_mean,
            Xhat_logvar,
            Z_mean,
            Z_logvar,
            A_mean,
            A_logvar
        )
        # Calculate total loss from different losses
        total_loss = negloglik_loss + self.beta * (kl_loss + self.att_beta * attkl_loss)
        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(negloglik_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.attkl_loss_tracker.update_state(attkl_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.rec_loss_tracker,
            self.kl_loss_tracker,
            self.attkl_loss_tracker,
        ]

    @tf.function
    def call(self, inputs, **kwargs):
        # Encoder is fed with input window
        Z_mean, Z_logvar, Z, states = self.encoder(inputs, training=False)
        # VS is fed with the hidden states from the encoder
        A_mean, A_logvar, A = self.vs(states, training=False)
        # Decoder is fed with the latent vector from the encoder and the attention matrix from VMS mechanism
        Xhat_mean, Xhat_logvar, Xhat = self.decoder([Z, A], training=False)
        return Xhat_mean, Xhat_logvar, Xhat, Z_mean, Z_logvar, Z, A_mean, A_logvar, A


class VSVAE_Encoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features, attn_vector_size, seed):
        super(VSVAE_Encoder, self).__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.attn_vector_size = attn_vector_size
        self.seed = seed
        self.encoder = self.build_encoder()

    def build_encoder(self):
        # Input window
        enc_input = tfkl.Input(shape=(self.seq_len, self.features))
        # Add small ammount of Gaussian noise
        enc_input = tfkl.GaussianNoise(0.1)(enc_input)
        #  BiLSTM layer
        bilstm = tfkl.Bidirectional(tfkl.LSTM(128, return_sequences=True))(enc_input)
        # L1 regularisation
        bilstm = tfkl.ActivityRegularization(l1=1e-8)(bilstm)
        # Take last hidden state
        last_states = bilstm[:, -1, :]
        # Transform deterministic BiLSTM output into distribution parameters Z_mean and Z_logvar
        Z_mean = tfkl.Dense(self.latent_dim, name="Z_mean")(last_states)
        Z_logvar = tfkl.Dense(self.latent_dim, name="Z_logvar")(last_states)
        # Create distribution object for reparametrisation trick
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        # Get epsilon for reparametrisation trick
        eps = output_dist.sample(tf.shape(Z_mean), seed=self.seed)
        # Reparametrisation trick
        Z = Z_mean + tf.sqrt(tf.math.exp(Z_logvar)) * eps
        return tf.keras.Model(enc_input, [Z_mean, Z_logvar, Z, bilstm], name="encoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.encoder(inputs, **kwargs)


class VSVAE_Decoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features, attn_vector_size, seed):
        super(VSVAE_Decoder, self).__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.attn_vector_size = attn_vector_size
        self.seed = seed
        self.decoder = self.build_decoder()

    def build_decoder(self):
        # Latent vector input
        latent_input = tfkl.Input(shape=(self.latent_dim,))
        # Repeat latent vector as many times as the window length (required for concatenation)
        latent_rep = tfkl.RepeatVector(self.seq_len)(latent_input)
        # Attention matrix input
        attention_input = tfkl.Input(shape=(self.seq_len, self.attn_vector_size))
        # Concatenate repeated latent vector and attention matrix
        input_dec = tfkl.Concatenate(axis=-1)([latent_rep, attention_input])
        #  BiLSTM layer
        bilstm = tfkl.Bidirectional(tfkl.LSTM(128, return_sequences=True))(input_dec)
        # L1 regularisation
        bilstm = tf.keras.layers.ActivityRegularization(l1=1e-8)(bilstm)
        # Transform deterministic BiLSTM output into distribution parameters Z_mean and Z_logvar
        Xhat_mean = tfkl.TimeDistributed(tfkl.Dense(self.features), name="Xhat_mean")(bilstm)
        Xhat_logvar = tfkl.TimeDistributed(tfkl.Dense(self.features), name="Xhat_logvar")(bilstm)
        # Create distribution object for reparametrisation trick
        output_dist = tfp.distributions.Laplace(loc=0., scale=1.)
        # Get epsilon for reparametrisation trick
        eps = output_dist.sample(tf.shape(Xhat_mean), seed=self.seed)
        # Reparametrisation trick
        Xhat = Xhat_mean + tf.sqrt(tf.math.exp(Xhat_logvar)) * eps
        return tf.keras.Model([latent_input, attention_input], [Xhat_mean, Xhat_logvar, Xhat], name="decoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.decoder(inputs, **kwargs)


class VS(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features, attn_vector_size, seed):
        super(VS, self).__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.attn_vector_size = attn_vector_size
        self.seed = seed
        self.vs = self.build_SA()

    def build_SA(self):
        # 256 is the dimensionality of the hidden states output by the encoder
        vs_input = tfkl.Input(shape=(self.seq_len, 256))
        # Get attention scores
        S_det = tf.divide(tf.matmul(vs_input, vs_input, transpose_b=True),
                          tf.sqrt(tf.cast(vs_input.shape[-1], 'float32')))
        # Multiply softmaxed attention scores (attention probabilities) with value matrix
        A_det = tf.matmul(tf.nn.softmax(S_det), vs_input)
        # Transform deterministic attention matrix into distribution parameters A_mean and A_logvar
        A_mean = tfkl.TimeDistributed(tfkl.Dense(self.attn_vector_size), name="A_mean")(A_det)
        A_logvar = tfkl.TimeDistributed(tfkl.Dense(self.attn_vector_size), name="A_logvar")(A_det)
        # Create distribution object for reparametrisation trick
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        # Get epsilon for reparametrisation trick
        eps = output_dist.sample(tf.shape(A_mean), seed=self.seed)
        # Reparametrisation trick
        A = A_mean + tf.sqrt(tf.math.exp(A_logvar)) * eps
        return tf.keras.Model(vs_input, [A_mean, A_logvar, A], name="VMSA")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.vs(inputs, **kwargs)


if __name__ == "__main__":
    pass
