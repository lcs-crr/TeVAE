"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands

Original paper DOI: 10.1504/IJDMB.2019.101395
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfkl = tf.keras.layers
tfd = tfp.distributions


class KLAnnealing(tf.keras.callbacks.Callback):
    def __init__(
            self,
            annealing_epochs: int = 25,
            annealing_type: str = "normal",
            grace_period: int = 0,
            beta_start: float = 1e-3,
            beta_end: float = 1e0
    ) -> None:
        super(KLAnnealing, self).__init__()
        self.annealing_epochs = annealing_epochs
        self.annealing_type = annealing_type
        self.grace_period = grace_period
        self.beta_start = beta_start
        self.beta_end = beta_end
        if annealing_type in ["cyclical", "monotonic"]:
            self.beta_values = tf.linspace(beta_start, beta_end, annealing_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        if self.annealing_type == "normal":
            new_value = self.beta_start
            self.model.beta.assign(new_value)
        elif self.annealing_type == "monotonic":
            if epoch < self.grace_period:
                new_value = self.beta_start
            elif self.grace_period <= epoch < self.grace_period + self.annealing_epochs:
                new_value = self.beta_values[epoch % self.annealing_epochs]
            else:
                new_value = self.beta_end
            self.model.beta.assign(new_value)
        elif self.annealing_type == "cyclical":
            if epoch < self.grace_period:
                new_value = self.beta_start
            else:
                new_value = self.beta_values[epoch % self.annealing_epochs]
            self.model.beta.assign(new_value)


class WVAE(tf.keras.Model):
    def __init__(
            self,
            encoder: tf.keras.Model,
            decoder: tf.keras.Model,
            beta: float = 1.0,
            name: str = None,
            **kwargs,
    ) -> None:
        super(WVAE, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta_tracker = tf.keras.metrics.Mean(name="beta")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.beta = tf.Variable(beta, trainable=False)  # Weight for KL-Loss, can be modified with a callback

    @staticmethod
    def rec_fn(x, xhat_params, reduce_time=True):
        xhat_mean, xhat_logvar = xhat_params
        # Configure distribution with output parameters
        output_dist = tfd.Laplace(loc=xhat_mean, scale=tf.sqrt(tf.math.exp(xhat_logvar)))
        # Calculate log probability of input data given output distribution
        loglik_loss = tf.reduce_sum(output_dist.log_prob(x), axis=-1)
        if reduce_time:
            return -tf.reduce_sum(loglik_loss, axis=1)
        else:
            return -loglik_loss

    @staticmethod
    def kldiv_fn(z_params):
        z_mean, z_logvar = z_params
        # Configure distribution with latent parameters
        latent_dist = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=tf.sqrt(tf.math.exp(z_logvar)))
        # Calculate KL-Divergence between latent distribution and standard Gaussian
        kl_loss = latent_dist.kl_divergence(
            tfd.MultivariateNormalDiag(loc=tf.zeros_like(z_mean), scale_diag=tf.ones_like(z_logvar))
        )
        return kl_loss

    def train_step(self, x, **kwargs):
        with tf.GradientTape() as tape:
            # Forward pass through encoder
            z_mean, z_logvar, z = self.encoder(x, training=True)
            # Forward pass through decoder
            xhat_mean, xhat_logvar, xhat = self.decoder(z, training=True)
            # Calculate losses from parameters
            rec_loss = self.rec_fn(x, [xhat_mean, xhat_logvar])
            kl_loss = self.kldiv_fn([z_mean, z_logvar])
            # Calculate total loss from different losses
            loss = rec_loss + self.beta * kl_loss
        # Calculate gradients in backward pass
        grads = tape.gradient(loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Track losses
        self.beta_tracker.update_state(self.beta)
        self.loss_tracker.update_state(loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "beta": self.beta_tracker.result(),
            "loss": self.loss_tracker.result(),
            "rec_loss": self.rec_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, x, **kwargs):
        # Forward pass through encoder
        z_mean, z_logvar, z = self.encoder(x, training=False)
        # Forward pass through decoder
        xhat_mean, xhat_logvar, xhat = self.decoder(z, training=False)
        # Calculate losses from parameters
        rec_loss = self.rec_fn(x, [xhat_mean, xhat_logvar])
        kl_loss = self.kldiv_fn([z_mean, z_logvar])
        loss = rec_loss + self.beta * kl_loss
        self.loss_tracker.update_state(loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics if m.name == 'rec_loss'}

    @property
    def metrics(self):
        return [
            self.beta_tracker,
            self.loss_tracker,
            self.rec_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def call(self, x, **kwargs):
        z_mean, z_logvar, z = self.encoder(x, training=False)
        xhat_mean, xhat_logvar, xhat = self.decoder(z, training=False)
        return xhat_mean, xhat_logvar, xhat, z_mean, z_logvar, z

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
            "beta": float(self.beta.numpy()),
        })
        return config

    @classmethod
    def from_config(cls, config, **kwargs):
        encoder = WVAE_Encoder.from_config(config["encoder"])
        decoder = WVAE_Decoder.from_config(config["decoder"])
        return cls(encoder=encoder, decoder=decoder, beta=config["beta"])


class WVAE_Encoder(tf.keras.Model):
    def __init__(
            self,
            seq_len: int,
            latent_dim: int,
            features: int,
            hidden_units: int,
            seed: int,
            name: str = None,
    ) -> None:
        super(WVAE_Encoder, self).__init__(name=name)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.hidden_units = hidden_units
        self.seed = seed
        self.encoder = self.build_encoder()

    def build_encoder(self):
        enc_input = tfkl.Input(shape=(self.seq_len, self.features))
        noised_input = tfkl.GaussianNoise(0.8, seed=self.seed)(enc_input)
        bilstm = tfkl.Bidirectional(tfkl.LSTM(self.hidden_units, return_sequences=False))(noised_input)
        bilstm = tfkl.ActivityRegularization(l1=1e-7)(bilstm)
        z_mean = tfkl.Dense(self.latent_dim)(bilstm)
        z_logvar = tfkl.Dense(self.latent_dim)(bilstm)
        eps = tf.random.normal(tf.shape(z_mean), seed=self.seed)
        z = z_mean + tf.sqrt(tf.math.exp(z_logvar)) * eps
        return tf.keras.Model(enc_input, [z_mean, z_logvar, z])

    @tf.function
    def call(self, x, **kwargs):
        return self.encoder(x, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "latent_dim": self.latent_dim,
            "features": self.features,
            "hidden_units": self.hidden_units,
            "seed": self.seed,
            "name": self.name,
        })
        return config

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(
            seq_len=config['seq_len'],
            latent_dim=config['latent_dim'],
            features=config['features'],
            hidden_units=config['hidden_units'],
            seed=config['seed'],
            name=config['name']
        )


class WVAE_Decoder(tf.keras.Model):
    def __init__(
            self,
            seq_len: int,
            latent_dim: int,
            features: int,
            hidden_units: int,
            seed: int,
            name: str = None,
    ) -> None:
        super(WVAE_Decoder, self).__init__(name=name)

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.hidden_units = hidden_units
        self.seed = seed
        self.decoder = self.build_decoder()

    def build_decoder(self):
        dec_input = tfkl.Input(shape=(self.latent_dim,))
        rep_input = tfkl.RepeatVector(self.seq_len)(dec_input)
        bilstm = tfkl.Bidirectional(tfkl.LSTM(self.hidden_units, return_sequences=True))(rep_input)
        bilstm = tf.keras.layers.ActivityRegularization(l1=1e-7)(bilstm)
        xhat_mean = tfkl.TimeDistributed(tfkl.Dense(self.features))(bilstm)
        xhat_logvar = tfkl.TimeDistributed(tfkl.Dense(self.features))(bilstm)
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        eps = output_dist.sample(tf.shape(xhat_mean), seed=self.seed)
        xhat = xhat_mean + tf.sqrt(tf.math.exp(xhat_logvar)) * eps
        return tf.keras.Model(dec_input, [xhat_mean, xhat_logvar, xhat])

    @tf.function
    def call(self, x, **kwargs):
        return self.decoder(x, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "latent_dim": self.latent_dim,
            "features": self.features,
            "hidden_units": self.hidden_units,
            "seed": self.seed,
            "name": self.name,
        })
        return config

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(
            seq_len=config['seq_len'],
            latent_dim=config['latent_dim'],
            features=config['features'],
            hidden_units=config['hidden_units'],
            seed=config['seed'],
            name=config['name']
        )


if __name__ == "__main__":
    pass
