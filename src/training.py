"""
Lucas Correia
Mercedes-Benz AG
Mercedesstr. 137 | 70327 Stuttgart | Germany
"""

import os
import tensorflow as tf

from models.vsvae import *
from models.omnianomaly import *
from models.wvae import *
from models.sisvae import *
from models.vasp import *
from models.lwvae import *
from models.noma import *
from models.tevae import *
from utils.kl_annealing import *

seed_list = [1, 2, 3]
data_split_list = ['1h', '8h', '64h', '512h']
max_epochs = 1

for seed in seed_list:
    tf.random.set_seed(seed)

    for data_split in data_split_list:
        data_load_path = 'Path to data'
        model_save_path = 'Path to model'

        # Load tf.data objects
        tfdata_train = tf.data.Dataset.load(os.path.join(data_load_path, data_split, 'train'))
        tfdata_val = tf.data.Dataset.load(os.path.join(data_load_path, data_split, 'val'))

        # Rebatch tf.data objects
        tfdata_train = tfdata_train.unbatch().batch(512)
        tfdata_val = tfdata_val.unbatch().batch(512)

        window_size = tfdata_train.element_spec.shape[1]
        features = tfdata_train.element_spec.shape[2]

        # Establish callbacks
        es = tf.keras.callbacks.EarlyStopping(monitor='val_rec_loss',
                                              mode='min',
                                              verbose=1,
                                              patience=250,
                                              restore_best_weights=True,
                                              )

        # region VSVAE
        encoder = VSVAE_Encoder(seq_len=window_size, latent_dim=3, features=features, attn_vector_size=3, seed=seed)
        decoder = VSVAE_Decoder(seq_len=window_size, latent_dim=3, features=features, attn_vector_size=3, seed=seed)
        vs = VS(seq_len=window_size, latent_dim=3, features=features, attn_vector_size=3, seed=seed)
        vsvae = VSVAE(encoder, decoder, vs, beta=1e-8, att_beta=1e-2)
        vsvae.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, clipnorm=10))

        annealing = KL_annealing(
            annealing_epochs=25,
            annealing_type="monotonic",
            grace_period=25,
            start=1e-8,
            end=1e-2,
        )

        # Fit model
        vsvae.fit(tfdata_train,
                  epochs=max_epochs,
                  callbacks=[annealing, es],
                  validation_data=tfdata_val,
                  verbose=2
                  )
        vsvae.predict(tf.random.normal((1, window_size, features)), verbose=0)
        vsvae.save(os.path.join(model_save_path, 'vsvae' + '_' + data_split + '_' + str(seed)))
        # endregion

        # region OMNIANOMALY
        encoder = OmniAnomaly_Encoder(seq_len=window_size, latent_dim=3, features=features, seed=seed)
        decoder = OmniAnomaly_Decoder(seq_len=window_size, latent_dim=3, features=features, seed=seed)
        omnianomaly = OmniAnomaly(encoder, decoder)
        omnianomaly.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True))

        # Fit vae model
        omnianomaly.fit(tfdata_train,
                        epochs=max_epochs,
                        callbacks=es,
                        validation_data=tfdata_val,
                        verbose=2
                        )
        omnianomaly.predict(tf.random.normal((1, window_size, features)), verbose=0)
        omnianomaly.save(os.path.join(model_save_path, 'omnianomaly' + '_' + data_split + '_' + str(seed)))
        # endregion

        # region WVAE
        encoder = WVAE_Encoder(seq_len=window_size, latent_dim=5, features=features, seed=seed)
        decoder = WVAE_Decoder(seq_len=window_size, latent_dim=5, features=features, seed=seed)
        wvae = WVAE(encoder, decoder, beta=1e-8)
        wvae.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, clipnorm=5))

        annealing = KL_annealing(
            annealing_epochs=25,
            annealing_type="monotonic",
            grace_period=25,
            start=1e-8,
            end=1e-2,
        )

        # Fit model
        wvae.fit(tfdata_train,
                 epochs=max_epochs,
                 callbacks=[es, annealing],
                 validation_data=tfdata_val,
                 verbose=2
                 )
        wvae.predict(tf.random.normal((1, window_size, features)), verbose=0)
        wvae.save(os.path.join(model_save_path, 'wvae' + '_' + data_split + '_' + str(seed)))
        # endregion

        # region SISVAE
        encoder = SISVAE_Encoder(seq_len=window_size, latent_dim=40, features=features, seed=seed)
        decoder = SISVAE_Decoder(seq_len=window_size, latent_dim=40, features=features, seed=seed)
        sisvae = SISVAE(encoder, decoder, beta=0.5)
        sisvae.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True))

        # Fit model
        sisvae.fit(tfdata_train,
                   epochs=max_epochs,
                   callbacks=es,
                   validation_data=tfdata_val,
                   verbose=2
                   )
        sisvae.predict(tf.random.normal((1, window_size, features)), verbose=0)
        sisvae.save(os.path.join(model_save_path, 'sisvae' + '_' + data_split + '_' + str(seed)))
        # endregion

        # region VASP
        encoder = VASP_Encoder(seq_len=window_size, latent_dim=8, features=features, seed=seed)
        decoder = VASP_Decoder(seq_len=window_size, latent_dim=8, features=features, seed=seed)
        vasp = VASP(encoder, decoder)
        vasp.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True))

        # Fit model
        vasp.fit(tfdata_train,
                 epochs=max_epochs,
                 callbacks=es,
                 validation_data=tfdata_val,
                 verbose=2
                 )
        vasp.predict(tf.random.normal((1, window_size, features)), verbose=0)
        vasp.save(os.path.join(model_save_path, 'vasp' + '_' + data_split + '_' + str(seed)))
        # endregion

        # region LWVAE
        encoder = LWVAE_Encoder(seq_len=window_size, latent_dim=64, features=features, seed=seed)
        decoder = LWVAE_Decoder(seq_len=window_size, latent_dim=64, features=features, seed=seed)
        lwvae = LWVAE(encoder, decoder)
        lwvae.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True))

        # Fit model
        lwvae.fit(tfdata_train,
                  epochs=max_epochs,
                  callbacks=es,
                  validation_data=tfdata_val,
                  verbose=2
                  )
        lwvae.predict(tf.random.normal((1, window_size, features)), verbose=0)
        lwvae.save(os.path.join(model_save_path, 'lwvae' + '_' + data_split + '_' + str(seed)))
        # endregion

        # region NoMA
        encoder = NoMA_Encoder(seq_len=window_size, latent_dim=64, features=features, seed=seed)
        decoder = NoMA_Decoder(seq_len=window_size, latent_dim=64, features=features, seed=seed)
        noma = NoMA(encoder, decoder, beta=1e-8)
        noma.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True))

        annealing = KL_annealing(
            annealing_epochs=25,
            annealing_type="cyclical",
            grace_period=25,
            start=1e-8,
            end=1e-2,
        )

        # Fit model
        noma.fit(tfdata_train,
                 epochs=max_epochs,
                 callbacks=[annealing, es],
                 validation_data=tfdata_val,
                 verbose=2
                 )
        noma.predict(tf.random.normal((1, window_size, features)), verbose=0)
        noma.save(os.path.join(model_save_path, 'noma' + '_' + data_split + '_' + str(seed)))
        # endregion

        # region TeVAE
        encoder = TeVAE_Encoder(seq_len=window_size, latent_dim=64, features=features, seed=seed)
        decoder = TeVAE_Decoder(seq_len=window_size, latent_dim=64, features=features, seed=seed)
        ma = MA(seq_len=window_size, latent_dim=64, features=features, key_dim=1)
        tevae = TeVAE(encoder, decoder, ma, beta=1e-8)
        tevae.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True))

        annealing = KL_annealing(
            annealing_epochs=25,
            annealing_type="cyclical",
            grace_period=25,
            start=1e-8,
            end=1e-2,
        )

        # Fit model
        tevae.fit(tfdata_train,
                  epochs=max_epochs,
                  callbacks=[annealing, es],
                  validation_data=tfdata_val,
                  verbose=2
                  )
        tevae.predict(tf.random.normal((1, window_size, features)), verbose=0)
        tevae.save(os.path.join(model_save_path, 'tevae' + '_' + data_split + '_' + str(seed)))
        # endregion

        tf.keras.backend.clear_session()
