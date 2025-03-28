"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands

Original paper DOI: 10.1109/TNNLS.2020.2980749
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfkl = tf.keras.layers
tfd = tfp.distributions


class SISVAE(tf.keras.Model):
    def __init__(
            self,
            encoder,
            decoder,
            name: str = None,
            **kwargs
    ) -> None:
        super(SISVAE, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.smooth_loss_tracker = tf.keras.metrics.Mean(name="smooth_loss")

    @staticmethod
    def rec_fn(x, xhat_params, reduce_time=True):
        xhat_mean, xhat_logvar = xhat_params
        # Configure distribution with output parameters
        output_dist = tfd.MultivariateNormalDiag(loc=xhat_mean, scale_diag=tf.sqrt(tf.math.exp(xhat_logvar)))
        # Calculate log probability of input data given output distribution
        loglik_loss = output_dist.log_prob(x)
        if reduce_time:
            return -tf.reduce_sum(loglik_loss, axis=1)
        else:
            return -loglik_loss

    @staticmethod
    def kldiv_fn(z_params, reduce_time=True):
        z_mean, z_logvar = z_params
        # Configure distribution with latent parameters
        latent_dist = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=tf.sqrt(tf.math.exp(z_logvar)))
        # Calculate KL-Divergence between latent distribution and standard Gaussian
        kl_loss = latent_dist.kl_divergence(
            tfd.MultivariateNormalDiag(loc=tf.zeros_like(z_mean), scale_diag=tf.ones_like(z_logvar))
        )
        if reduce_time:
            return tf.reduce_sum(kl_loss, axis=1)
        else:
            return kl_loss

    @staticmethod
    def smooth_fn(z_params, reduce_time=True):
        z_mean, z_logvar = z_params
        # Calculate KL Divergence between t-1 latent distribution and current latent distribution
        smooth_loss = [
            tfd.MultivariateNormalDiag(loc=z_mean[:, time_step - 1], scale_diag=tf.sqrt(tf.math.exp(z_logvar[:, time_step - 1]))).kl_divergence(
                tfd.MultivariateNormalDiag(loc=z_mean[:, time_step], scale_diag=tf.sqrt(tf.math.exp(z_logvar[:, time_step]))),
            ) for time_step in range(1, z_mean.shape[1])
        ]
        smooth_loss = tf.transpose(tf.stack(smooth_loss), perm=[1, 0])
        if reduce_time:
            return tf.reduce_sum(smooth_loss, axis=1)
        else:
            return smooth_loss

    def train_step(self, x, **kwargs):
        with tf.GradientTape() as tape:
            # Forward pass through encoder
            z_mean, z_logvar, z = self.encoder(x, training=True)
            # Forward pass through decoder
            xhat_mean, xhat_logvar, xhat = self.decoder(z, training=True)
            # Calculate losses from parameters
            rec_loss = self.rec_fn(x, [xhat_mean, xhat_logvar])
            kl_loss = self.kldiv_fn([z_mean, z_logvar])
            smooth_loss = self.smooth_fn([z_mean, z_logvar])
            # Calculate total loss from different losses
            loss = rec_loss + kl_loss + 0.5 * smooth_loss
        # Calculate gradients in backward pass
        grads = tape.gradient(loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Track losses
        self.loss_tracker.update_state(loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.smooth_loss_tracker.update_state(smooth_loss)
        return {
            "loss": self.loss_tracker.result(),
            "rec_loss": self.rec_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "smooth_loss": self.smooth_loss_tracker.result(),
        }

    def test_step(self, x, **kwargs):
        # Forward pass through encoder
        z_mean, z_logvar, z = self.encoder(x, training=False)
        # Forward pass through decoder
        xhat_mean, xhat_logvar, xhat = self.decoder(z, training=False)
        # Calculate losses from parameters
        rec_loss = self.rec_fn(x, [xhat_mean, xhat_logvar])
        kl_loss = self.kldiv_fn([z_mean, z_logvar])
        smooth_loss = self.smooth_fn([z_mean, z_logvar])
        # Calculate total loss from different losses
        loss = rec_loss + kl_loss + 0.5 * smooth_loss
        self.loss_tracker.update_state(loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.smooth_loss_tracker.update_state(smooth_loss)
        return {m.name: m.result() for m in self.metrics if m.name == 'rec_loss'}

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.rec_loss_tracker,
            self.kl_loss_tracker,
            self.smooth_loss_tracker,
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
        })
        return config

    @classmethod
    def from_config(cls, config, **kwargs):
        encoder = SISVAE_Encoder.from_config(config["encoder"])
        decoder = SISVAE_Decoder.from_config(config["decoder"])
        return cls(encoder=encoder, decoder=decoder)


class SISVAE_Encoder(tf.keras.Model):
    def __init__(
            self,
            seq_len: int,
            latent_dim: int,
            features: int,
            hidden_units: int,
            seed: int,
            name: str = None,
    ) -> None:
        super(SISVAE_Encoder, self).__init__(name=name)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.hidden_units = hidden_units
        self.seed = seed
        self.encoder = self.build_encoder()

    def build_encoder(self):
        enc_input = tfkl.Input(shape=(self.seq_len, self.features))
        gru = tfkl.GRU(self.hidden_units, return_sequences=True)(enc_input)
        z_mean = tfkl.TimeDistributed(tfkl.Dense(self.latent_dim))(gru)
        z_logvar = tfkl.TimeDistributed(tfkl.Dense(self.latent_dim))(gru)
        eps = tf.random.normal(tf.shape(z_mean), seed=self.seed)
        z = z_mean + tf.sqrt(tf.math.exp(z_logvar)) * eps
        return tf.keras.Model(enc_input, [z_mean, z_logvar, z],)

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


class SISVAE_Decoder(tf.keras.Model):
    def __init__(
            self,
            seq_len: int,
            latent_dim: int,
            features: int,
            hidden_units: int,
            seed: int,
            name: str = None,
    ) -> None:
        super(SISVAE_Decoder, self).__init__(name=name)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.hidden_units = hidden_units
        self.seed = seed
        self.decoder = self.build_decoder()

    def build_decoder(self):
        latent_input = tfkl.Input(shape=(self.seq_len, self.latent_dim,))
        gru = tfkl.GRU(self.hidden_units, return_sequences=True)(latent_input)
        xhat_mean = tfkl.TimeDistributed(tfkl.Dense(self.features))(gru)
        xhat_logvar = tfkl.TimeDistributed(tfkl.Dense(self.features))(gru)
        eps = tf.random.normal(tf.shape(xhat_mean), seed=self.seed)
        xhat = xhat_mean + tf.sqrt(tf.math.exp(xhat_logvar)) * eps
        return tf.keras.Model(latent_input, [xhat_mean, xhat_logvar, xhat])

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
