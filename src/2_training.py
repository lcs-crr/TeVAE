"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from dotenv import dotenv_values
from utilities import data_class

# Load variables in .env file
config = dotenv_values("../.env")

# Load directory paths from .env file
data_path = config['data_path']
model_path = config['model_path']

# Declare constants
MODEL_NAME = 'tevae'  # or 'omnianomaly', 'sisvae', 'lwvae', 'vsvae', 'vasp', 'wvae', 'noma'

data_split_list = ['1h', '8h', '64h', '512h']

for seed in range(1, 4):
    # Set fixed seed for random operations
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    for data_split in data_split_list:
        # Declare model name and paths
        data_load_path = os.path.join(data_path, '2_preprocessed')
        model_save_path = os.path.join(model_path, MODEL_NAME + '_' + data_split + '_' + str(seed))

        # Load data
        tfdata_train = tf.data.Dataset.load(os.path.join(data_load_path, data_split, 'train'))
        tfdata_val = tf.data.Dataset.load(os.path.join(data_load_path, data_split, 'val'))

        tfdata_train = tfdata_train.cache().batch(512).prefetch(tf.data.AUTOTUNE)
        tfdata_val = tfdata_val.cache().batch(512).prefetch(tf.data.AUTOTUNE)

        # Establish callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_rec_loss',
            mode='min',
            verbose=1,
            patience=250,
            restore_best_weights=True,
        )

        # Define model
        window_size = tfdata_train.element_spec.shape[1]
        features = tfdata_train.element_spec.shape[2]
        if MODEL_NAME == 'tevae':
            from model_garden.tevae import *
            annealing = KLAnnealing(
                annealing_type="cyclical",
                beta_start=1e-8,
                beta_end=1e-2
            )
            latent_dim = 64
            key_dim = 1
            hidden_units = 256
            encoder = TEVAE_Encoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            decoder = TEVAE_Decoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            ma = MA(seq_len=window_size, latent_dim=latent_dim, key_dim=key_dim, features=features)
            model = TEVAE(encoder, decoder, ma)
            callback_list = [early_stopping, annealing]
            model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, clipnorm=None))

        elif MODEL_NAME == 'omnianomaly':
            from model_garden.omnianomaly import *
            latent_dim = 3
            hidden_units = 500
            encoder = OmniAnomaly_Encoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            decoder = OmniAnomaly_Decoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            model = OmniAnomaly(encoder, decoder)
            callback_list = [early_stopping]
            model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=False, clipnorm=10))

        elif MODEL_NAME == 'sisvae':
            from model_garden.sisvae import *
            latent_dim = 40
            hidden_units = 200
            encoder = SISVAE_Encoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            decoder = SISVAE_Decoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            model = SISVAE(encoder, decoder)
            callback_list = [early_stopping]
            model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=False, clipnorm=None))

        elif MODEL_NAME == 'lwvae':
            from model_garden.lwvae import *
            latent_dim = 64
            hidden_units = 128
            encoder = LWVAE_Encoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            decoder = LWVAE_Decoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            model = LWVAE(encoder, decoder)
            callback_list = [early_stopping]
            model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=False, clipnorm=None))

        elif MODEL_NAME == 'vsvae':
            from model_garden.vsvae import *
            annealing = KLAnnealing(
                annealing_type="monotonic",
                beta_start=1e-8,
                beta_end=1e-2
            )
            latent_dim = 3
            hidden_units = 128
            encoder = VSVAE_Encoder(seq_len=window_size, latent_dim=latent_dim, features=features, hidden_units=hidden_units, seed=seed)
            decoder = VSVAE_Decoder(seq_len=window_size, latent_dim=latent_dim, features=features, hidden_units=hidden_units, seed=seed)
            vs = VS(seq_len=window_size, latent_dim=latent_dim, features=features, seed=seed)
            model = VSVAE(encoder, decoder, vs)
            callback_list = [early_stopping, annealing]
            model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, clipnorm=1))

        elif MODEL_NAME == 'vasp':
            from model_garden.vasp import *
            latent_dim = 8
            hidden_units = 65
            encoder = VASP_Encoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            decoder = VASP_Decoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            model = VASP(encoder, decoder)
            callback_list = [early_stopping]
            model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=False, clipnorm=None))

        elif MODEL_NAME == 'wvae':
            from model_garden.wvae import *
            annealing = KLAnnealing(
                annealing_type="monotonic",
                beta_start=1e-8,
                beta_end=1e-2
            )
            latent_dim = 5
            hidden_units = 128
            encoder = WVAE_Encoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            decoder = WVAE_Decoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            model = WVAE(encoder, decoder)
            callback_list = [early_stopping, annealing]
            model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, clipnorm=5))

        elif MODEL_NAME == 'noma':
            from model_garden.noma import *
            annealing = KLAnnealing(
                annealing_type="cyclical",
                beta_start=1e-8,
                beta_end=1e-2
            )
            latent_dim = 64
            hidden_units = 256
            encoder = NOMA_Encoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            decoder = NOMA_Decoder(seq_len=window_size, latent_dim=latent_dim, hidden_units=hidden_units, features=features, seed=seed)
            model = NOMA(encoder, decoder)
            callback_list = [early_stopping, annealing]
            model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, clipnorm=None))

        # Fit model
        history = model.fit(tfdata_train,
                            epochs=10000,
                            callbacks=callback_list,
                            validation_data=tfdata_val,
                            verbose=2
                            )

        # Run model on random data with same shape as input to build model
        model.predict(tf.random.normal((32, window_size, features)), verbose=0)

        # Save model and losses
        model.save(model_save_path)
        data_class.DataProcessor().dump_pickle(history, os.path.join(model_save_path, 'losses.pkl'))
        tf.keras.backend.clear_session()
