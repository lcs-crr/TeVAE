"""
Lucas Correia
Mercedes-Benz AG
Mercedesstr. 137 | 70327 Stuttgart | Germany
"""

import os
import numpy as np
import pickle
import random
import tensorflow as tf
import ts_processor
import ts_plotter
from scipy import integrate
import sklearn.metrics
import scipy
import utility_functions

seed_list = [1, 2, 3]
data_split_list = ['1h', '8h', '64h', '512h']
reverse_window_mode = 'mean'
# reverse_window_mode = 'last'
# reverse_window_mode = 'first'
sampling_rate = 2
model = 'tevae'

data_load_path = 'Path to data'
model_load_path = 'Path to model'

for model_seed in model_seed_list:
    for data_stand in data_stand_list:
        seed = model_seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        channel_list = ['Vehicle Speed [-]',
                        'EDU Torque [-]',
                        'Left Axle Torque [-]',
                        'Right Axle Torque [-]',
                        'EDU Current [-]',
                        'EDU Voltage [-]',
                        'HVB Current [-]',
                        'HVB Voltage [-]',
                        'HVB Temperature [-]',
                        'HVB State of Charge [-]',
                        'EDU Rotor Temperature [-]',
                        'EDU Stator Temperature [-]',
                        'Inverter Temperature [-]',
                        ]

        val_list = utility_functions.load_pickle(os.path.join(data_load_path, 'val.pkl'))
        test_list = utility_functions.load_pickle(os.path.join(data_load_path, 'test.pkl'))

        model = []
        if os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'val_detection_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'val_rootcause_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'val_recon_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'val_latent_' + rev_mode + '.pkl')):
            val_detection_score = utility_functions.load_pickle(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'val_detection_score_' + rev_mode + '.pkl'))

            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'val_rootcause_score_' + rev_mode + '.pkl'), 'rb') as f:
                val_rootcause_score = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'val_recon_' + rev_mode + '.pkl'), 'rb') as f:
                val_recon = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'val_latent_' + rev_mode + '.pkl'), 'rb') as f:
                val_latent = pickle.load(f)
            print('Finished loading validation inference results.')
        else:
            model = tf.keras.models.load_model(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed)))
            print('Finished loading model.')
            val_detection_score = []
            val_rootcause_score = []
            val_recon = []
            val_latent = []
            for time_series in val_data:
                detection_score, rootcause_score, recon, latent = ts_processor.inference_s_vae(model, time_series, rev_mode, window_size)
                val_detection_score.append(detection_score)
                val_rootcause_score.append(rootcause_score)
                val_recon.append(recon)
                val_latent.append(latent)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'val_detection_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(val_detection_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'val_rootcause_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(val_rootcause_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'val_recon_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(val_recon, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'val_latent_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(val_latent, f)
            print('Finished inference on validation data.')

        # Process normal test data
        if os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'test_normal_detection_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'test_normal_rootcause_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'test_normal_recon_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'test_normal_latent_' + rev_mode + '.pkl')):
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'test_normal_detection_score_' + rev_mode + '.pkl'), 'rb') as f:
                test_normal_detection_score = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'test_normal_rootcause_score_' + rev_mode + '.pkl'), 'rb') as f:
                test_normal_rootcause_score = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'test_normal_recon_' + rev_mode + '.pkl'), 'rb') as f:
                test_normal_recon = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'test_normal_latent_' + rev_mode + '.pkl'), 'rb') as f:
                test_normal_latent = pickle.load(f)
            print('Finished loading normal inference results.')
        else:
            if not model:
                model = tf.keras.models.load_model(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed)))
                print('Finished loading model.')
            test_normal_detection_score = []
            test_normal_rootcause_score = []
            test_normal_recon = []
            test_normal_latent = []
            for time_series in test_normal:
                detection_score, rootcause_score, recon, latent = ts_processor.inference_s_vae(model, time_series, rev_mode, window_size)
                test_normal_detection_score.append(detection_score)
                test_normal_rootcause_score.append(rootcause_score)
                test_normal_recon.append(recon)
                test_normal_latent.append(latent)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'test_normal_detection_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(test_normal_detection_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'test_normal_rootcause_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(test_normal_rootcause_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'test_normal_recon_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(test_normal_recon, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'test_normal_latent_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(test_normal_latent, f)
            print('Finished inference on normal data.')

        # Process motor pump middle data
        if os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_middle_detection_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_middle_rootcause_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_middle_recon_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_middle_latent_' + rev_mode + '.pkl')):
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_middle_detection_score_' + rev_mode + '.pkl'), 'rb') as f:
                motor_pump_middle_detection_score = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_middle_rootcause_score_' + rev_mode + '.pkl'), 'rb') as f:
                motor_pump_middle_rootcause_score = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_middle_recon_' + rev_mode + '.pkl'), 'rb') as f:
                motor_pump_middle_recon = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_middle_latent_' + rev_mode + '.pkl'), 'rb') as f:
                motor_pump_middle_latent = pickle.load(f)
            print('Finished loading motor pump middle inference results.')
        else:
            if not model:
                model = tf.keras.models.load_model(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed)))
                print('Finished loading model.')
            motor_pump_middle_detection_score = []
            motor_pump_middle_rootcause_score = []
            motor_pump_middle_recon = []
            motor_pump_middle_latent = []
            for time_series in motor_pump_middle:
                detection_score, rootcause_score, recon, latent = ts_processor.inference_s_vae(model, time_series, rev_mode, window_size)
                motor_pump_middle_detection_score.append(detection_score)
                motor_pump_middle_rootcause_score.append(rootcause_score)
                motor_pump_middle_recon.append(recon)
                motor_pump_middle_latent.append(latent)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_middle_detection_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(motor_pump_middle_detection_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_middle_rootcause_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(motor_pump_middle_rootcause_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_middle_recon_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(motor_pump_middle_recon, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_middle_latent_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(motor_pump_middle_latent, f)
            print('Finished inference on motor pump middle data.')

        # Process motor pump beginning data
        if os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_beginning_detection_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_beginning_rootcause_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_beginning_recon_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_beginning_latent_' + rev_mode + '.pkl')):
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_beginning_detection_score_' + rev_mode + '.pkl'), 'rb') as f:
                motor_pump_beginning_detection_score = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_beginning_rootcause_score_' + rev_mode + '.pkl'), 'rb') as f:
                motor_pump_beginning_rootcause_score = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_beginning_recon_' + rev_mode + '.pkl'), 'rb') as f:
                motor_pump_beginning_recon = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_beginning_latent_' + rev_mode + '.pkl'), 'rb') as f:
                motor_pump_beginning_latent = pickle.load(f)
            print('Finished loading motor pump beginning inference results.')
        else:
            if not model:
                model = tf.keras.models.load_model(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed)))
                print('Finished loading model.')
            motor_pump_beginning_detection_score = []
            motor_pump_beginning_rootcause_score = []
            motor_pump_beginning_recon = []
            motor_pump_beginning_latent = []
            for time_series in motor_pump_beginning:
                detection_score, rootcause_score, recon, latent = ts_processor.inference_s_vae(model, time_series, rev_mode, window_size)
                motor_pump_beginning_detection_score.append(detection_score)
                motor_pump_beginning_rootcause_score.append(rootcause_score)
                motor_pump_beginning_recon.append(recon)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_beginning_detection_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(motor_pump_beginning_detection_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_beginning_rootcause_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(motor_pump_beginning_rootcause_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_beginning_recon_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(motor_pump_beginning_recon, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'motor_pump_beginning_latent_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(motor_pump_beginning_latent, f)
            print('Finished inference on motor pump beginning data.')

        # Process wheel diameter data
        if os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'wheel_diameter_detection_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'wheel_diameter_rootcause_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'wheel_diameter_recon_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'wheel_diameter_latent_' + rev_mode + '.pkl')):
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'wheel_diameter_detection_score_' + rev_mode + '.pkl'), 'rb') as f:
                wheel_diameter_detection_score = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'wheel_diameter_rootcause_score_' + rev_mode + '.pkl'), 'rb') as f:
                wheel_diameter_rootcause_score = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'wheel_diameter_recon_' + rev_mode + '.pkl'), 'rb') as f:
                wheel_diameter_recon = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'wheel_diameter_latent_' + rev_mode + '.pkl'), 'rb') as f:
                wheel_diameter_latent = pickle.load(f)
            print('Finished loading wheel diameter inference results.')
        else:
            if not model:
                model = tf.keras.models.load_model(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed)))
                print('Finished loading model.')
            wheel_diameter_detection_score = []
            wheel_diameter_rootcause_score = []
            wheel_diameter_recon = []
            wheel_diameter_latent = []
            for time_series in wheel_diameter:
                detection_score, rootcause_score, recon, latent = ts_processor.inference_s_vae(model, time_series, rev_mode, window_size)
                wheel_diameter_detection_score.append(detection_score)
                wheel_diameter_rootcause_score.append(rootcause_score)
                wheel_diameter_recon.append(recon)
                wheel_diameter_latent.append(latent)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'wheel_diameter_detection_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(wheel_diameter_detection_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'wheel_diameter_rootcause_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(wheel_diameter_rootcause_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'wheel_diameter_recon_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(wheel_diameter_recon, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'wheel_diameter_latent_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(wheel_diameter_latent, f)
            print('Finished inference on wheel diameter data.')

        # Process recup off data
        if os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'recup_off_detection_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'recup_off_rootcause_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'recup_off_recon_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'recup_off_latent_' + rev_mode + '.pkl')):
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'recup_off_detection_score_' + rev_mode + '.pkl'), 'rb') as f:
                recup_off_detection_score = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'recup_off_rootcause_score_' + rev_mode + '.pkl'), 'rb') as f:
                recup_off_rootcause_score = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'recup_off_recon_' + rev_mode + '.pkl'), 'rb') as f:
                recup_off_recon = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'recup_off_latent_' + rev_mode + '.pkl'), 'rb') as f:
                recup_off_latent = pickle.load(f)
            print('Finished loading recuperation off inference results.')
        else:
            if not model:
                model = tf.keras.models.load_model(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed)))
                print('Finished loading model.')
            recup_off_detection_score = []
            recup_off_rootcause_score = []
            recup_off_recon = []
            recup_off_latent = []
            for time_series in recup_off:
                detection_score, rootcause_score, recon, latent = ts_processor.inference_s_vae(model, time_series, rev_mode, window_size)
                recup_off_detection_score.append(detection_score)
                recup_off_rootcause_score.append(rootcause_score)
                recup_off_recon.append(recon)
                recup_off_latent.append(latent)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'recup_off_detection_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(recup_off_detection_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'recup_off_rootcause_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(recup_off_rootcause_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'recup_off_recon_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(recup_off_recon, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'recup_off_latent_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(recup_off_latent, f)
            print('Finished inference on recuperation off data.')

        # Process batt sim data
        if os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'batt_sim_detection_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'batt_sim_rootcause_score_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'batt_sim_recon_' + rev_mode + '.pkl')) \
                and os.path.isfile(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'batt_sim_latent_' + rev_mode + '.pkl')):
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'batt_sim_detection_score_' + rev_mode + '.pkl'), 'rb') as f:
                batt_sim_detection_score = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'batt_sim_rootcause_score_' + rev_mode + '.pkl'), 'rb') as f:
                batt_sim_rootcause_score = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'batt_sim_recon_' + rev_mode + '.pkl'), 'rb') as f:
                batt_sim_recon = pickle.load(f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'batt_sim_latent_' + rev_mode + '.pkl'), 'rb') as f:
                batt_sim_latent = pickle.load(f)
            print('Finished loading battery simulator inference results.')
        else:
            if not model:
                model = tf.keras.models.load_model(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed)))
                print('Finished loading model.')
            batt_sim_detection_score = []
            batt_sim_rootcause_score = []
            batt_sim_recon = []
            batt_sim_latent = []
            for time_series in batt_sim:
                detection_score, rootcause_score, recon, latent = ts_processor.inference_s_vae(model, time_series, rev_mode, window_size)
                batt_sim_detection_score.append(detection_score)
                batt_sim_rootcause_score.append(rootcause_score)
                batt_sim_recon.append(recon)
                batt_sim_latent.append(latent)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'batt_sim_detection_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(batt_sim_detection_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'batt_sim_rootcause_score_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(batt_sim_rootcause_score, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'batt_sim_recon_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(batt_sim_recon, f)
            with open(os.path.join(model_load_path, model_folder + '_' + data_stand + '_' + str(model_seed), 'batt_sim_latent_' + rev_mode + '.pkl'), 'wb') as f:
                pickle.dump(batt_sim_latent, f)
            print('Finished inference on battery simulator data.')
        print()
        # endregion

        window_size = tfdata_train.element_spec.shape[1]

        # region Evaluation section
        # Evaluate validation data to obtain threshold
        red_val_data_error = []
        for i, score in enumerate(val_detection_score):
            # score = score * np.mean(val_recon[i][1], axis=1, keepdims=True)
            # score = np.prod(val_rootcause_score[i], axis=1, keepdims=True)
            if red_mode == 'percentile':
                red_val_data_error.append(np.percentile(score, percentile))
            elif red_mode == 'mean':
                red_val_data_error.append(np.mean(score))
            elif red_mode == 'std':
                red_val_data_error.append(np.std(score))
            elif red_mode == 'area':
                red_val_data_error.append(integrate.simpson(score[:, 0]))
        red_val_data_error = np.vstack(red_val_data_error)
        threshold = np.percentile(red_val_data_error, percentile)

        total_delays = []

        # Evaluate normal test data to obtain TN and FP
        FP_normal = 0
        TN_normal = 0
        red_test_normal_detection_score = []
        test_normal_rootcause_channels = []
        for i, score in enumerate(test_normal_detection_score):
            if np.percentile(score, percentile) >= threshold:
                FP_normal += 1
                test_normal_rootcause_channels.append(np.nan)
            else:
                TN_normal += 1
            red_test_normal_detection_score.append(np.percentile(score, percentile))
        test_normal_rootcause_labels = [False] * len(test_normal_rootcause_channels)
        test_normal_groundtruth_labels = [True] * len(test_normal_rootcause_channels)

        TP_motor_pump_middle = 0
        FN_motor_pump_middle = 0
        FP_motor_pump_middle = 0
        red_motor_pump_middle_detection_score = []
        motor_pump_middle_rootcause_channels = []
        motor_pump_middle_groundtruth_channels = [10, 11, 12]
        anomaly_start = 0.5
        for i, score in enumerate(motor_pump_middle_detection_score):
            # Is anomaly predicted?
            if np.percentile(score, percentile) >= threshold:  # Yes
                gt = np.zeros_like(score)
                gt[len(gt) // 2:] = 1
                # Does predicted anomaly overlap with ground truth?
                if np.sum(np.logical_and(score >= threshold, gt)) > 0:  # Yes
                    _, uncorrected_delay = ts_processor.find_detection_delay(score, threshold, sampling_rate, rev_mode, window_size, len(score), anomaly_start)
                    # Does prediction start before ground truth?
                    if uncorrected_delay < len(score) * anomaly_start:  # Yes
                        FP_motor_pump_middle += 1
                        delay = len(score) / sampling_rate
                        total_delays.append(delay)
                        motor_pump_middle_rootcause_channels.append(np.nan)
                    else:  # No
                        TP_motor_pump_middle += 1
                        delay, _ = ts_processor.find_detection_delay(score, threshold, sampling_rate, rev_mode, window_size, len(score), anomaly_start)
                        total_delays.append(delay)
                        motor_pump_middle_rootcause_channels.append(np.argmax([motor_pump_middle_rootcause_score[i][np.argmax(score[:, 0] > threshold), j] for j in range(len(channel_list))]))
                else:  # No
                    FP_motor_pump_middle += 1
                    delay = len(score) / sampling_rate
                    total_delays.append(delay)
                    motor_pump_middle_rootcause_channels.append(np.nan)
            else:  # No
                FN_motor_pump_middle += 1
                delay = len(score) / sampling_rate
                total_delays.append(delay)
            red_motor_pump_middle_detection_score.append(np.percentile(score, percentile))
        motor_pump_middle_rootcause_labels = [x in motor_pump_middle_groundtruth_channels for x in motor_pump_middle_rootcause_channels]
        motor_pump_middle_groundtruth_labels = [True] * len(motor_pump_middle_rootcause_labels)

        TP_motor_pump_beginning = 0
        FN_motor_pump_beginning = 0
        red_motor_pump_beginning_detection_score = []
        motor_pump_beginning_rootcause_channels = []
        motor_pump_beginning_groundtruth_channels = [10, 11, 12]
        anomaly_start = 0
        for i, score in enumerate(motor_pump_beginning_detection_score):
            if np.percentile(score, percentile) >= threshold:
                TP_motor_pump_beginning += 1
                delay, _ = ts_processor.find_detection_delay(score, threshold, sampling_rate, rev_mode, window_size, len(score), anomaly_start)
                total_delays.append(delay)
                motor_pump_beginning_rootcause_channels.append(np.argmax([motor_pump_beginning_rootcause_score[i][np.argmax(score[:, 0] > threshold), j] for j in range(len(channel_list))]))
            else:
                FN_motor_pump_beginning += 1
                delay = len(score) / sampling_rate
                total_delays.append(delay)
            red_motor_pump_beginning_detection_score.append(np.percentile(score, percentile))
        motor_pump_beginning_rootcause_labels = [x in motor_pump_beginning_groundtruth_channels for x in motor_pump_beginning_rootcause_channels]
        motor_pump_beginning_groundtruth_labels = [True] * len(motor_pump_beginning_rootcause_labels)

        TP_wheel_diameter = 0
        FN_wheel_diameter = 0
        red_wheel_diameter_detection_score = []
        wheel_diameter_rootcause_channels = []
        wheel_diameter_groundtruth_channels = [0]
        anomaly_start = 0
        for i, score in enumerate(wheel_diameter_detection_score):
            if np.percentile(score, percentile) >= threshold:
                TP_wheel_diameter += 1
                delay, _ = ts_processor.find_detection_delay(score, threshold, sampling_rate, rev_mode, window_size, len(score), anomaly_start)
                total_delays.append(delay)
                wheel_diameter_rootcause_channels.append(np.argmax([wheel_diameter_rootcause_score[i][np.argmax(score[:, 0] > threshold), j] for j in range(len(channel_list))]))
            else:
                FN_wheel_diameter += 1
                delay = len(score) / sampling_rate
                total_delays.append(delay)
            red_wheel_diameter_detection_score.append(np.percentile(score, percentile))
        wheel_diameter_rootcause_labels = [x in wheel_diameter_groundtruth_channels for x in wheel_diameter_rootcause_channels]
        wheel_diameter_groundtruth_labels = [True] * len(wheel_diameter_rootcause_labels)

        TP_recup_off = 0
        FN_recup_off = 0
        red_recup_off_detection_score = []
        recup_off_rootcause_channels = []
        recup_off_groundtruth_channels = [1, 2, 3, 9]
        anomaly_start = 0
        for i, score in enumerate(recup_off_detection_score):
            if np.percentile(score, percentile) >= threshold:
                TP_recup_off += 1
                delay, _ = ts_processor.find_detection_delay(score, threshold, sampling_rate, rev_mode, window_size, len(score), anomaly_start)
                total_delays.append(delay)
                recup_off_rootcause_channels.append(np.argmax([recup_off_rootcause_score[i][np.argmax(score[:, 0] > threshold), j] for j in range(len(channel_list))]))
            else:
                FN_recup_off += 1
                delay = len(score) / sampling_rate
                total_delays.append(delay)
            red_recup_off_detection_score.append(np.percentile(score, percentile))
        recup_off_rootcause_labels = [x in recup_off_groundtruth_channels for x in recup_off_rootcause_channels]
        recup_off_groundtruth_labels = [True] * len(recup_off_rootcause_labels)

        TP_batt_sim = 0
        FN_batt_sim = 0
        red_batt_sim_detection_score = []
        batt_sim_rootcause_channels = []
        batt_sim_groundtruth_channels = [6, 7, 8, 9]
        anomaly_start = 0
        for i, score in enumerate(batt_sim_detection_score):
            if np.percentile(score, percentile) >= threshold:
                TP_batt_sim += 1
                delay, _ = ts_processor.find_detection_delay(score, threshold, sampling_rate, rev_mode, window_size, len(score), anomaly_start)
                total_delays.append(delay)
                batt_sim_rootcause_channels.append(np.argmax([batt_sim_rootcause_score[i][np.argmax(score[:, 0] > threshold), j] for j in range(len(channel_list))]))
            else:
                FN_batt_sim += 1
                delay = len(score) / sampling_rate
                total_delays.append(delay)
            red_batt_sim_detection_score.append(np.percentile(score, percentile))
        batt_sim_rootcause_labels = [x in batt_sim_groundtruth_channels for x in batt_sim_rootcause_channels]
        batt_sim_groundtruth_labels = [True] * len(batt_sim_rootcause_labels)
        print('Finished evaluating test data.')

        FP = FP_normal + FP_motor_pump_middle
        TN = TN_normal
        TP = TP_motor_pump_middle + TP_motor_pump_beginning + TP_wheel_diameter + TP_recup_off + TP_batt_sim
        FN = FN_motor_pump_middle + FN_motor_pump_beginning + FN_wheel_diameter + FN_recup_off + FN_batt_sim

        from sklearn import metrics

        precision_gs = []
        recall_gs = []
        f1_gs = []
        percentile_array = np.arange(0, 100.1, 0.1)
        score_normal = red_test_normal_detection_score
        score_anomaly = red_motor_pump_middle_detection_score + red_motor_pump_beginning_detection_score + red_wheel_diameter_detection_score + red_recup_off_detection_score + red_batt_sim_detection_score
        for percentile in percentile_array:
            threshold_temp = np.percentile(np.concatenate((score_normal, score_anomaly)), percentile)
            predicted_labels_normal = score_normal >= threshold_temp
            predicted_labels_anomaly = score_anomaly >= threshold_temp
            predicted_labels_all = np.concatenate((predicted_labels_normal, predicted_labels_anomaly), axis=0)
            true_labels_normal = np.zeros_like(predicted_labels_normal)
            true_labels_anomaly = np.ones_like(predicted_labels_anomaly)
            true_labels_all = np.concatenate((true_labels_normal, true_labels_anomaly), axis=0)
            precision = metrics.precision_score(true_labels_all, predicted_labels_all)
            recall = metrics.recall_score(true_labels_all, predicted_labels_all)
            f1 = metrics.f1_score(true_labels_all, predicted_labels_all)
            precision_gs.append(precision)
            recall_gs.append(recall)
            f1_gs.append(f1)
        precision_gs = np.vstack(precision_gs)
        recall_gs = np.vstack(recall_gs)
        f1_gs = np.vstack(f1_gs)
        auc = metrics.auc(recall_gs[:, 0], precision_gs[:, 0])
        threshold_best = np.percentile(np.concatenate((score_normal, score_anomaly)), percentile_array[np.argmax(f1_gs)])

        assert TP + TN + FP + FN == len(score_normal) + len(score_anomaly)

        TP_rc = np.sum(motor_pump_middle_rootcause_labels + motor_pump_beginning_rootcause_labels + wheel_diameter_rootcause_labels + recup_off_rootcause_labels + batt_sim_rootcause_labels)
        FP_rc = np.sum(np.logical_not(motor_pump_middle_rootcause_labels + motor_pump_beginning_rootcause_labels + wheel_diameter_rootcause_labels + recup_off_rootcause_labels + batt_sim_rootcause_labels + test_normal_rootcause_labels))

        assert TP + FP == TP_rc + FP_rc

        print()
        print(model_folder)
        print(data_stand)
        print(model_seed)
        print('Unsupervised performance:')
        print('TP:')
        print(TP)
        print('FN:')
        print(FN)
        print('TN:')
        print(TN)
        print('FP:')
        print(FP)
        print('Key Metrics:')
        print(TP / (TP + FP))
        print(TP / (TP + FN))
        print(TP / (TP + 0.5 * (FP + FN)))
        print(auc)
        print(np.mean(total_delays))
        print(TP_rc / (FP_rc + TP_rc))
        print()

        threshold = threshold_best
        total_delays = []

        # Evaluate normal test data to obtain TN and FP
        FP_normal = 0
        TN_normal = 0
        red_test_normal_detection_score = []
        test_normal_rootcause_channels = []
        for i, score in enumerate(test_normal_detection_score):
            if np.percentile(score, percentile) >= threshold:
                FP_normal += 1
                test_normal_rootcause_channels.append(np.nan)
            else:
                TN_normal += 1
            red_test_normal_detection_score.append(np.percentile(score, percentile))
        test_normal_rootcause_labels = [False] * len(test_normal_rootcause_channels)
        test_normal_groundtruth_labels = [True] * len(test_normal_rootcause_channels)

        TP_motor_pump_middle = 0
        FN_motor_pump_middle = 0
        FP_motor_pump_middle = 0
        red_motor_pump_middle_detection_score = []
        motor_pump_middle_rootcause_channels = []
        motor_pump_middle_groundtruth_channels = [10, 11, 12]
        anomaly_start = 0.5
        for i, score in enumerate(motor_pump_middle_detection_score):
            # Is anomaly predicted?
            if np.percentile(score, percentile) >= threshold:  # Yes
                gt = np.zeros_like(score)
                gt[len(gt) // 2:] = 1
                # Does predicted anomaly overlap with ground truth?
                if np.sum(np.logical_and(score >= threshold, gt)) > 0:  # Yes
                    _, uncorrected_delay = ts_processor.find_detection_delay(score, threshold, sampling_rate, rev_mode, window_size, len(score), anomaly_start)
                    # Does prediction start before ground truth?
                    if uncorrected_delay < len(score) * anomaly_start:  # Yes
                        FP_motor_pump_middle += 1
                        delay = len(score) / sampling_rate
                        total_delays.append(delay)
                        motor_pump_middle_rootcause_channels.append(np.nan)
                    else:  # No
                        TP_motor_pump_middle += 1
                        delay, _ = ts_processor.find_detection_delay(score, threshold, sampling_rate, rev_mode, window_size, len(score), anomaly_start)
                        total_delays.append(delay)
                        motor_pump_middle_rootcause_channels.append(np.argmax([motor_pump_middle_rootcause_score[i][np.argmax(score[:, 0] > threshold), j] for j in range(len(channel_list))]))
                else:  # No
                    FP_motor_pump_middle += 1
                    delay = len(score) / sampling_rate
                    total_delays.append(delay)
                    motor_pump_middle_rootcause_channels.append(np.nan)
            else:  # No
                FN_motor_pump_middle += 1
                delay = len(score) / sampling_rate
                total_delays.append(delay)
            red_motor_pump_middle_detection_score.append(np.percentile(score, percentile))
        motor_pump_middle_rootcause_labels = [x in motor_pump_middle_groundtruth_channels for x in motor_pump_middle_rootcause_channels]
        motor_pump_middle_groundtruth_labels = [True] * len(motor_pump_middle_rootcause_labels)

        TP_motor_pump_beginning = 0
        FN_motor_pump_beginning = 0
        red_motor_pump_beginning_detection_score = []
        motor_pump_beginning_rootcause_channels = []
        motor_pump_beginning_groundtruth_channels = [10, 11, 12]
        anomaly_start = 0
        for i, score in enumerate(motor_pump_beginning_detection_score):
            if np.percentile(score, percentile) >= threshold:
                TP_motor_pump_beginning += 1
                delay, _ = ts_processor.find_detection_delay(score, threshold, sampling_rate, rev_mode, window_size, len(score), anomaly_start)
                total_delays.append(delay)
                motor_pump_beginning_rootcause_channels.append(np.argmax([motor_pump_beginning_rootcause_score[i][np.argmax(score[:, 0] > threshold), j] for j in range(len(channel_list))]))
            else:
                FN_motor_pump_beginning += 1
                delay = len(score) / sampling_rate
                total_delays.append(delay)
            red_motor_pump_beginning_detection_score.append(np.percentile(score, percentile))
        motor_pump_beginning_rootcause_labels = [x in motor_pump_beginning_groundtruth_channels for x in motor_pump_beginning_rootcause_channels]
        motor_pump_beginning_groundtruth_labels = [True] * len(motor_pump_beginning_rootcause_labels)

        TP_wheel_diameter = 0
        FN_wheel_diameter = 0
        red_wheel_diameter_detection_score = []
        wheel_diameter_rootcause_channels = []
        wheel_diameter_groundtruth_channels = [0]
        anomaly_start = 0
        for i, score in enumerate(wheel_diameter_detection_score):
            if np.percentile(score, percentile) >= threshold:
                TP_wheel_diameter += 1
                delay, _ = ts_processor.find_detection_delay(score, threshold, sampling_rate, rev_mode, window_size, len(score), anomaly_start)
                total_delays.append(delay)
                wheel_diameter_rootcause_channels.append(np.argmax([wheel_diameter_rootcause_score[i][np.argmax(score[:, 0] > threshold), j] for j in range(len(channel_list))]))
            else:
                FN_wheel_diameter += 1
                delay = len(score) / sampling_rate
                total_delays.append(delay)
            red_wheel_diameter_detection_score.append(np.percentile(score, percentile))
        wheel_diameter_rootcause_labels = [x in wheel_diameter_groundtruth_channels for x in wheel_diameter_rootcause_channels]
        wheel_diameter_groundtruth_labels = [True] * len(wheel_diameter_rootcause_labels)

        TP_recup_off = 0
        FN_recup_off = 0
        red_recup_off_detection_score = []
        recup_off_rootcause_channels = []
        recup_off_groundtruth_channels = [1, 2, 3, 9]
        anomaly_start = 0
        for i, score in enumerate(recup_off_detection_score):
            if np.percentile(score, percentile) >= threshold:
                TP_recup_off += 1
                delay, _ = ts_processor.find_detection_delay(score, threshold, sampling_rate, rev_mode, window_size, len(score), anomaly_start)
                total_delays.append(delay)
                recup_off_rootcause_channels.append(np.argmax([recup_off_rootcause_score[i][np.argmax(score[:, 0] > threshold), j] for j in range(len(channel_list))]))
            else:
                FN_recup_off += 1
                delay = len(score) / sampling_rate
                total_delays.append(delay)
            red_recup_off_detection_score.append(np.percentile(score, percentile))
        recup_off_rootcause_labels = [x in recup_off_groundtruth_channels for x in recup_off_rootcause_channels]
        recup_off_groundtruth_labels = [True] * len(recup_off_rootcause_labels)

        TP_batt_sim = 0
        FN_batt_sim = 0
        red_batt_sim_detection_score = []
        batt_sim_rootcause_channels = []
        batt_sim_groundtruth_channels = [6, 7, 8, 9]
        anomaly_start = 0
        for i, score in enumerate(batt_sim_detection_score):
            if np.percentile(score, percentile) >= threshold:
                TP_batt_sim += 1
                delay, _ = ts_processor.find_detection_delay(score, threshold, sampling_rate, rev_mode, window_size, len(score), anomaly_start)
                total_delays.append(delay)
                batt_sim_rootcause_channels.append(np.argmax([batt_sim_rootcause_score[i][np.argmax(score[:, 0] > threshold), j] for j in range(len(channel_list))]))
            else:
                FN_batt_sim += 1
                delay = len(score) / sampling_rate
                total_delays.append(delay)
            red_batt_sim_detection_score.append(np.percentile(score, percentile))
        batt_sim_rootcause_labels = [x in batt_sim_groundtruth_channels for x in batt_sim_rootcause_channels]
        batt_sim_groundtruth_labels = [True] * len(batt_sim_rootcause_labels)

        FP = FP_normal + FP_motor_pump_middle
        TN = TN_normal
        TP = TP_motor_pump_middle + TP_motor_pump_beginning + TP_wheel_diameter + TP_recup_off + TP_batt_sim
        FN = FN_motor_pump_middle + FN_motor_pump_beginning + FN_wheel_diameter + FN_recup_off + FN_batt_sim

        assert TP + TN + FP + FN == len(score_normal) + len(score_anomaly)

        TP_rc = np.sum(motor_pump_middle_rootcause_labels + motor_pump_beginning_rootcause_labels + wheel_diameter_rootcause_labels + recup_off_rootcause_labels + batt_sim_rootcause_labels)
        FP_rc = np.sum(np.logical_not(motor_pump_middle_rootcause_labels + motor_pump_beginning_rootcause_labels + wheel_diameter_rootcause_labels + recup_off_rootcause_labels + batt_sim_rootcause_labels + test_normal_rootcause_labels))

        assert TP + FP == TP_rc + FP_rc

        print(model_folder)
        print(data_stand)
        print(model_seed)
        print('Theoretical maximum performance:')
        print('TP:')
        print(TP)
        print('FN:')
        print(FN)
        print('TN:')
        print(TN)
        print('FP:')
        print(FP)
        print('Key Metrics:')
        print(TP / (TP + FP))
        print(TP / (TP + FN))
        print(TP / (TP + 0.5 * (FP + FN)))
        print(auc)
        print(np.mean(total_delays))
        print(TP_rc / (FP_rc + TP_rc))
        print()
        print('Finished evaluation.')
        print()
        print()

        # endregion

        tf.keras.backend.clear_session()
    print()
print()
