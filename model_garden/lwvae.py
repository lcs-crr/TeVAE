"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands

Original paper DOI: 10.3390/s22082886
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfkl = tf.keras.layers
tfd = tfp.distributions


@tf.keras.saving.register_keras_serializable(package="LWVAE")
class LWVAE(tf.keras.Model):
    def __init__(
            self,
            encoder: tf.keras.Model,
            decoder: tf.keras.Model,
            name: str = None,
            **kwargs
    ) -> None:
        super(LWVAE, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @staticmethod
    def rec_fn(x, x_hat, reduce_time=True):
        if reduce_time:
            return tf.reduce_sum(tf.losses.MeanSquaredError('none')(x, x_hat), axis=1)
        else:
            return tf.losses.MeanSquaredError('none')(x, x_hat)

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
            xhat = self.decoder(z, training=True)
            # Calculate losses from parameters
            rec_loss = self.rec_fn(x, xhat)
            kl_loss = self.kldiv_fn([z_mean, z_logvar])
            # Calculate total loss from different losses
            loss = rec_loss + kl_loss
        # Calculate gradients in backward pass
        grads = tape.gradient(loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Track losses
        self.loss_tracker.update_state(loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.loss_tracker.result(),
            "rec_loss": self.rec_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, x, **kwargs):
        # Forward pass through encoder
        z_mean, z_logvar, z = self.encoder(x, training=False)
        # Forward pass through decoder
        xhat = self.decoder(z, training=False)
        # Calculate losses from parameters
        rec_loss = self.rec_fn(x, xhat)
        kl_loss = self.kldiv_fn([z_mean, z_logvar])
        # Calculate total loss from different losses
        loss = rec_loss + kl_loss
        self.loss_tracker.update_state(loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics if m.name == 'rec_loss'}

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.rec_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def call(self, x, **kwargs):
        z_mean, z_logvar, z = self.encoder(x, training=False)
        xhat = self.decoder(z, training=False)
        return xhat, z_mean, z_logvar, z

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
        })
        return config

    @classmethod
    def from_config(cls, config, **kwargs):
        encoder = LWVAE_Encoder.from_config(config["encoder"])
        decoder = LWVAE_Decoder.from_config(config["decoder"])
        return cls(encoder=encoder, decoder=decoder)


@tf.keras.saving.register_keras_serializable(package="LWVAE")
class LWVAE_Encoder(tf.keras.Model):
    def __init__(
            self,
            seq_len: int,
            latent_dim: int,
            features: int,
            hidden_units: int,
            seed: int,
            name: str = None,
    ) -> None:
        super(LWVAE_Encoder, self).__init__(name=name)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.hidden_units = hidden_units
        self.seed = seed
        self.encoder = self.build_encoder()

    def build_encoder(self):
        enc_input = tfkl.Input(shape=(self.seq_len, self.features))
        lstm = tfkl.LSTM(self.hidden_units, return_sequences=False)(enc_input)
        z_mean = tfkl.Dense(self.latent_dim)(lstm)
        z_logvar = tfkl.Dense(self.latent_dim)(lstm)
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


@tf.keras.saving.register_keras_serializable(package="LWVAE")
class LWVAE_Decoder(tf.keras.Model):
    def __init__(
            self,
            seq_len: int,
            latent_dim: int,
            features: int,
            hidden_units: int,
            seed: int,
            name: str = None,
    ) -> None:
        super(LWVAE_Decoder, self).__init__(name=name)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.hidden_units = hidden_units
        self.seed = seed
        self.decoder = self.build_decoder()

    def build_decoder(self):
        latent_input = tfkl.Input(shape=(self.latent_dim,))
        repeat_latent = tfkl.RepeatVector(self.seq_len)(latent_input)
        lstm = tfkl.LSTM(self.hidden_units, return_sequences=True)(repeat_latent)
        xhat = tfkl.TimeDistributed(tfkl.Dense(self.features))(lstm)
        return tf.keras.Model(latent_input, xhat)

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