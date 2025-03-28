"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands

Original paper DOI: 10.1145/3292500.3330672
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfkl = tf.keras.layers
tfd = tfp.distributions


@tf.keras.saving.register_keras_serializable(package="OmniAnomaly")
class OmniAnomaly(tf.keras.Model):
    def __init__(
            self,
            encoder: tf.keras.Model,
            decoder: tf.keras.Model,
            name: str = None,
            **kwargs
    ) -> None:
        super(OmniAnomaly, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.lgssm_loss_tracker = tf.keras.metrics.Mean(name="lgssm_loss")

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
    def lgssm_fn(z):
        # Configure distribution with output parameters
        lgssm_dist = tfd.LinearGaussianStateSpaceModel(
            num_timesteps=z.shape[1],
            transition_matrix=tf.linalg.LinearOperatorIdentity(z.shape[-1]),
            transition_noise=tfd.MultivariateNormalDiag(scale_diag=tf.ones([z.shape[-1]])),
            observation_matrix=tf.linalg.LinearOperatorIdentity(z.shape[-1]),
            observation_noise=tfd.MultivariateNormalDiag(scale_diag=tf.ones([z.shape[-1]])),
            initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=tf.ones([z.shape[-1]]))
        )
        # Calculate log probability of sample belongs parametrised distribution
        return -lgssm_dist.log_prob(z)

    def train_step(self, x, **kwargs):
        with tf.GradientTape() as tape:
            # Forward pass through encoder
            z_mean, z_logvar, z = self.encoder(x, training=True)
            # Forward pass through decoder
            xhat_mean, xhat_logvar, xhat = self.decoder(z, training=True)
            # Calculate losses from parameters
            rec_loss = self.rec_fn(x, [xhat_mean, xhat_logvar])
            kl_loss = self.kldiv_fn([z_mean, z_logvar])
            lgssm_loss = self.lgssm_fn(z)
            # Calculate total loss from different losses
            loss = rec_loss + kl_loss + lgssm_loss
        # Calculate gradients in backward pass
        grads = tape.gradient(loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Track losses
        self.loss_tracker.update_state(loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.lgssm_loss_tracker.update_state(lgssm_loss)
        return {
            "loss": self.loss_tracker.result(),
            "rec_loss": self.rec_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "lgssm_loss": self.lgssm_loss_tracker.result(),
        }

    def test_step(self, x, **kwargs):
        # Forward pass through encoder
        z_mean, z_logvar, z = self.encoder(x, training=False)
        # Forward pass through decoder
        xhat_mean, xhat_logvar, xhat = self.decoder(z, training=False)
        # Calculate losses from parameters
        rec_loss = self.rec_fn(x, [xhat_mean, xhat_logvar])
        kl_loss = self.kldiv_fn([z_mean, z_logvar])
        lgssm_loss = self.lgssm_fn(z)
        # Calculate total loss from different losses
        loss = rec_loss + kl_loss + lgssm_loss
        self.loss_tracker.update_state(loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.lgssm_loss_tracker.update_state(lgssm_loss)
        return {m.name: m.result() for m in self.metrics if m.name == 'rec_loss'}

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.rec_loss_tracker,
            self.kl_loss_tracker,
            self.lgssm_loss_tracker,
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
        encoder = OmniAnomaly_Encoder.from_config(config["encoder"])
        decoder = OmniAnomaly_Decoder.from_config(config["decoder"])
        return cls(encoder=encoder, decoder=decoder)


@tf.keras.saving.register_keras_serializable(package="OmniAnomaly")
class OmniAnomaly_Encoder(tf.keras.Model):
    def __init__(
            self,
            seq_len: int,
            latent_dim: int,
            features: int,
            hidden_units: int,
            seed: int,
            name: str = None,
    ) -> None:
        super(OmniAnomaly_Encoder, self).__init__(name=name)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.hidden_units = hidden_units
        self.seed = seed
        self.encoder = self.build_encoder()

    def build_encoder(self):
        enc_input = tfkl.Input(shape=(self.seq_len, self.features))
        x = tfkl.GRU(self.hidden_units, return_sequences=True, activity_regularizer=tf.keras.regularizers.L2(l2=1e-4))(enc_input)
        h = tfkl.TimeDistributed(tfkl.Dense(self.hidden_units, activation="relu", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(x)
        z_mean = tfkl.TimeDistributed(tfkl.Dense(self.latent_dim, activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(h)
        z_logvar = tfkl.TimeDistributed(tfkl.Dense(self.latent_dim, activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(h)
        eps = tf.random.normal(tf.shape(z_mean), seed=self.seed)
        z = z_mean + tf.sqrt(tf.math.exp(z_logvar)) * eps + 1e-4
        for k in range(20):
            z = z + tfkl.Dense(self.latent_dim, use_bias=False)(tfkl.Dense(self.latent_dim, activation='tanh')(z))
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


@tf.keras.saving.register_keras_serializable(package="OmniAnomaly")
class OmniAnomaly_Decoder(tf.keras.Model):
    def __init__(
            self,
            seq_len: int,
            latent_dim: int,
            features: int,
            hidden_units: int,
            seed: int,
            name: str = None,
    ) -> None:
        super(OmniAnomaly_Decoder, self).__init__(name=name)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.hidden_units = hidden_units
        self.seed = seed
        self.decoder = self.build_decoder()

    def build_decoder(self):
        dec_input = tfkl.Input(shape=(self.seq_len, self.latent_dim))
        x = tfkl.GRU(self.hidden_units, return_sequences=True, activity_regularizer=tf.keras.regularizers.L2(l2=1e-4))(dec_input)
        h = tfkl.TimeDistributed(tfkl.Dense(self.hidden_units, activation="relu", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(x)
        xhat_mean = tfkl.TimeDistributed(tfkl.Dense(self.features, activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(h)
        xhat_logvar = tfkl.TimeDistributed(tfkl.Dense(self.features, activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(h)
        eps = tf.random.normal(tf.shape(xhat_mean), seed=self.seed)
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
