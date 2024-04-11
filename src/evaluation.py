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
import sklearn.metrics
from scipy import integrate
import utility_functions

seed_list = [1, 2, 3]
data_split_list = ['1h', '8h', '64h', '512h']
reverse_window_mode = 'mean'
# reverse_window_mode = 'last'
# reverse_window_mode = 'first'
sampling_rate = 2
model_name = 'tevae'
window_size = 256

data_load_path = 'Path to data'
model_load_path = 'Path to model'

for model_seed in seed_list:
    for data_split in data_split_list:
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

        model_config = model_name + '_' + data_split + '_' + str(model_seed)

        # Inference on validation data
        model = tf.keras.models.load_model(os.path.join(model_load_path, model_config))
        val_anomaly_score = []
        val_rootcause_score = []
        val_recon = []
        val_latent = []
        for val_ts in val_list:
            if model_name == 'vasp':
                anomaly_score, rootcause_score, recon, latent = utility_functions.inference_vasp(model, time_series, rev_mode, window_size)
            elif model_name == 'lwvae':
                anomaly_score, rootcause_score, recon, latent = utility_functions.inference_lwvae(model, time_series, rev_mode, window_size)
            else:
                anomaly_score, rootcause_score, recon, latent = utility_functions.inference_vae(model, time_series, rev_mode, window_size)
            val_anomaly_score.append(anomaly_score)
            val_rootcause_score.append(rootcause_score)
            val_recon.append(recon)
            val_latent.append(latent)
        utility_functions.dump_pickle(os.path.join(model_load_path, model_config, 'val_anomaly_score_' + rev_mode + '.pkl'), val_anomaly_score)
        utility_functions.dump_pickle(os.path.join(model_load_path, model_config, 'val_rootcause_score_' + rev_mode + '.pkl'), val_rootcause_score)
        utility_functions.dump_pickle(os.path.join(model_load_path, model_config, 'val_recon_' + rev_mode + '.pkl'), val_recon)
        utility_functions.dump_pickle(os.path.join(model_load_path, model_config, 'val_latent_' + rev_mode + '.pkl'), val_latent)

        # Inference on test data
        test_anomaly_score = []
        test_rootcause_score = []
        test_recon = []
        test_latent = []
        for test_ts in test_list:
            if model_name == 'vasp':
                anomaly_score, rootcause_score, recon, latent = utility_functions.inference_vasp(model, time_series, rev_mode, window_size)
            elif model_name == 'lwvae':
                anomaly_score, rootcause_score, recon, latent = utility_functions.inference_lwvae(model, time_series, rev_mode, window_size)
            else:
                anomaly_score, rootcause_score, recon, latent = utility_functions.inference_vae(model, time_series, rev_mode, window_size)
            test_anomaly_score.append(anomaly_score)
            test_rootcause_score.append(rootcause_score)
            test_recon.append(recon)
            test_latent.append(latent)
        utility_functions.dump_pickle(os.path.join(model_load_path, model_config, 'test_anomaly_score_' + rev_mode + '.pkl'), test_anomaly_score)
        utility_functions.dump_pickle(os.path.join(model_load_path, model_config, 'test_rootcause_score_' + rev_mode + '.pkl'), test_rootcause_score)
        utility_functions.dump_pickle(os.path.join(model_load_path, model_config, 'test_recon_' + rev_mode + '.pkl'), test_recon)
        utility_functions.dump_pickle(os.path.join(model_load_path, model_config, 'test_latent_' + rev_mode + '.pkl'), test_latent)

        # Evaluate validation data to obtain threshold
        max_val_list_error = [np.max(score_ts) for score_ts in val_anomaly_score]
        threshold = np.percentile(red_val_data_error, 100)

        motor_pump_middle_groundtruth_rootcause_channels = [10, 11, 12]
        motor_pump_beginning_groundtruth_rootcause_channels = [10, 11, 12]
        wheel_diameter_groundtruth_rootcause_channels = [0]
        recup_off_groundtruth_rootcause_channels = [1, 2, 3, 9]
        batt_sim_groundtruth_rootcause_channels = [6, 7, 8, 9]

        # Evaluate test data using unsupervised threshold
        FP = 0
        TN = 0
        FN = 0
        TP = 0
        total_delays = []
        total_test_anomaly_score = []
        test_rootcause_channels = []
        test_rootcause_labels = []
        for idx_test, score_ts in enumerate(test_anomaly_score):
            label = test_list[idx_test].dtype.metadata['label']
            # Ground-truth normal time series
            if label == 'normal':
                # >0 time steps in anomaly score higher than threshold
                # False positive
                if np.sum(score_ts >= threshold) > 0:
                    FP += 1
                    test_rootcause_channels.append(np.nan)
                    test_rootcause_labels.append(label)
                # =0 time steps in anomaly score higher than threshold
                # True negative
                else:
                    TN += 1

            # Ground-truth anomalous time series
            else:
                # Extract groundtruth anomaly start from file name
                if label == 'motor_pump_middle':
                    groundtruth_anomaly_start = 0.5 * len(score_ts)
                else:
                    groundtruth_anomaly_start = 0
                # >0 time steps in anomaly score higher than threshold
                # Anomaly predicted
                if np.sum(score_ts >= threshold) > 0:
                    predicted_anomaly_start = np.argwhere(score_ts >= threshold)[0][0]
                    # First predicted anomalous time step is after the groundtruth anomaly start
                    # True positive
                    if predicted_anomaly_start > groundtruth_anomaly_start:
                        TP += 1
                        delay, _ = ts_processor.find_detection_delay(score_ts, threshold, sampling_rate, reverse_window_mode, window_size, len(score_ts), groundtruth_anomaly_start)
                        total_delays.append(delay)
                        test_rootcause_channels.append(np.argmax([test_rootcause_score[idx_test][np.argmax(score_ts[:, 0] > threshold), j] for j in range(len(channel_list))]))
                        test_rootcause_labels.append(label)
                    # First predicted anomalous time step is before the groundtruth anomaly start
                    # False positive
                    else:
                        FP += 1
                        delay = len(score_ts) / sampling_rate
                        total_delays.append(delay)
                        test_rootcause_channels.append(np.nan)
                        test_rootcause_labels.append(label)
                # =0 time steps in anomaly score higher than threshold
                # False negative
                else:
                    FN += 1
                    delay = len(score_ts) / sampling_rate
                    total_delays.append(delay)
            # Append to list with all test anomaly scores
            total_test_anomaly_score.append(score_ts)

        TP_rc = 0
        FP_rc = 0
        for idx_test, _ in enumerate(test_rootcause_channels):
            if test_rootcause_labels[idx_test].dtype.metadata['label'] == 'motor_pump_middle':
                if test_rootcause_channels[idx_test] in motor_pump_middle_groundtruth_rootcause_channels:
                    TP_rc += 1
                else:
                    FP_rc += 1
            elif test_rootcause_labels[idx_test].dtype.metadata['label'] == 'motor_pump_beginning':
                if test_rootcause_channels[idx_test] in motor_pump_beginning_groundtruth_rootcause_channels:
                    TP_rc += 1
                else:
                    FP_rc += 1
            elif test_rootcause_labels[idx_test].dtype.metadata['label'] == 'wheel_diameter':
                if test_rootcause_channels[idx_test] in wheel_diameter_groundtruth_rootcause_channels:
                    TP_rc += 1
                else:
                    FP_rc += 1
            elif test_rootcause_labels[idx_test].dtype.metadata['label'] == 'recup_off':
                if test_rootcause_channels[idx_test] in recup_off_groundtruth_rootcause_channels:
                    TP_rc += 1
                else:
                    FP_rc += 1
            elif test_rootcause_labels[idx_test].dtype.metadata['label'] == 'batt_sim':
                if test_rootcause_channels[idx_test] in batt_sim_groundtruth_rootcause_channels:
                    TP_rc += 1
                else:
                    FP_rc += 1

        precision_list = []
        recall_list = []
        f1_list = []
        flattened_test_anomaly_score = np.concatenate(total_test_anomaly_score).ravel()
        percentile_array = np.arange(0, 100.1, 0.1)
        for threshold_percentile in percentile_array:
            threshold_temp = np.percentile(flattened_test_anomaly_score, threshold_percentile)
            groundtruth_labels = []
            predicted_labels = []
            for idx_test, score_ts in enumerate(test_anomaly_score):
                label = test_list[idx_test].dtype.metadata['label']
                # Ground-truth normal time series
                if label == 'normal':
                    # >0 time steps in anomaly score higher than threshold
                    # False positive
                    if np.sum(score_ts >= threshold_temp) > 0:
                        predicted_labels.append(True)
                        groundtruth_labels.append(False)
                    # =0 time steps in anomaly score higher than threshold
                    # True negative
                    else:
                        predicted_labels.append(False)
                        groundtruth_labels.append(False)
                # Ground-truth anomalous time series
                else:
                    # Extract groundtruth anomaly start from file name
                    if label == 'motor_pump_middle':
                        groundtruth_anomaly_start = 0.5 * len(score_ts)
                    else:
                        groundtruth_anomaly_start = 0
                    # >0 time steps in anomaly score higher than threshold
                    # Anomaly predicted
                    if np.sum(score_ts >= threshold_temp) > 0:
                        predicted_anomaly_start = np.argwhere(score_ts >= threshold_temp)[0][0]
                        # First predicted anomalous time step is after the groundtruth anomaly start
                        # True positive
                        if predicted_anomaly_start > groundtruth_anomaly_start:
                            predicted_labels.append(True)
                            groundtruth_labels.append(True)
                        # First predicted anomalous time step is before the groundtruth anomaly start
                        # False positive
                        else:
                            predicted_labels.append(True)
                            groundtruth_labels.append(False)
                    # =0 time steps in anomaly score higher than threshold
                    # False negative
                    else:
                        predicted_labels.append(False)
                        groundtruth_labels.append(True)
            precision = metrics.precision_score(groundtruth_labels, predicted_labels)
            recall = metrics.recall_score(groundtruth_labels, predicted_labels)
            f1 = metrics.f1_score(groundtruth_labels, predicted_labels)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
        precision_list = np.vstack(precision_list)
        recall_list = np.vstack(recall_list)
        f1_list = np.vstack(f1_list)
        a_pr = metrics.auc(recall_list[:, 0], precision_list[:, 0])
        threshold_best = np.percentile(flattened_test_anomaly_score, percentile_array[np.argmax(f1_list)])

        print()
        print(model_name)
        print(data_split)
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
        print('Precision:')
        print(TP / (TP + FP))
        print('Recall:')
        print(TP / (TP + FN))
        print('F1 score:')
        print(TP / (TP + 0.5 * (FP + FN)))
        print('Area under precision-recall curve:')
        print(a_pr)
        print('Average detection delay in seconds:')
        print(np.mean(total_delays))
        print('Rootcause precision:')
        print(TP_rc / (FP_rc + TP_rc))
        print()

        threshold = threshold_best
        
        # Reevaluate test data using ideal threshold
        FP = 0
        TN = 0
        FN = 0
        TP = 0
        total_delays = []
        total_test_anomaly_score = []
        test_rootcause_channels = []
        test_rootcause_labels = []
        for idx_test, score_ts in enumerate(test_anomaly_score):
            label = test_list[idx_test].dtype.metadata['label']
            # Ground-truth normal time series
            if label == 'normal':
                # >0 time steps in anomaly score higher than threshold
                # False positive
                if np.sum(score_ts >= threshold) > 0:
                    FP += 1
                    test_rootcause_channels.append(np.nan)
                    test_rootcause_labels.append(label)
                # =0 time steps in anomaly score higher than threshold
                # True negative
                else:
                    TN += 1

            # Ground-truth anomalous time series
            else:
                # Extract groundtruth anomaly start from file name
                if label == 'motor_pump_middle':
                    groundtruth_anomaly_start = 0.5 * len(score_ts)
                else:
                    groundtruth_anomaly_start = 0
                # >0 time steps in anomaly score higher than threshold
                # Anomaly predicted
                if np.sum(score_ts >= threshold) > 0:
                    predicted_anomaly_start = np.argwhere(score_ts >= threshold)[0][0]
                    # First predicted anomalous time step is after the groundtruth anomaly start
                    # True positive
                    if predicted_anomaly_start > groundtruth_anomaly_start:
                        TP += 1
                        delay, _ = ts_processor.find_detection_delay(score_ts, threshold, sampling_rate, reverse_window_mode, window_size, len(score_ts), groundtruth_anomaly_start)
                        total_delays.append(delay)
                        test_rootcause_channels.append(np.argmax([test_rootcause_score[idx_test][np.argmax(score_ts[:, 0] > threshold), j] for j in range(len(channel_list))]))
                        test_rootcause_labels.append(label)
                    # First predicted anomalous time step is before the groundtruth anomaly start
                    # False positive
                    else:
                        FP += 1
                        delay = len(score_ts) / sampling_rate
                        total_delays.append(delay)
                        test_rootcause_channels.append(np.nan)
                        test_rootcause_labels.append(label)
                # =0 time steps in anomaly score higher than threshold
                # False negative
                else:
                    FN += 1
                    delay = len(score_ts) / sampling_rate
                    total_delays.append(delay)
            # Append to list with all test anomaly scores
            total_test_anomaly_score.append(score_ts)
            
        TP_rc = 0
        FP_rc = 0
        for idx_test, _ in enumerate(test_rootcause_channels):
            if test_rootcause_labels[idx_test].dtype.metadata['label'] == 'motor_pump_middle':
                if test_rootcause_channels[idx_test] in motor_pump_middle_groundtruth_rootcause_channels:
                    TP_rc += 1
                else:
                    FP_rc += 1
            elif test_rootcause_labels[idx_test].dtype.metadata['label'] == 'motor_pump_beginning':
                if test_rootcause_channels[idx_test] in motor_pump_beginning_groundtruth_rootcause_channels:
                    TP_rc += 1
                else:
                    FP_rc += 1
            elif test_rootcause_labels[idx_test].dtype.metadata['label'] == 'wheel_diameter':
                if test_rootcause_channels[idx_test] in wheel_diameter_groundtruth_rootcause_channels:
                    TP_rc += 1
                else:
                    FP_rc += 1
            elif test_rootcause_labels[idx_test].dtype.metadata['label'] == 'recup_off':
                if test_rootcause_channels[idx_test] in recup_off_groundtruth_rootcause_channels:
                    TP_rc += 1
                else:
                    FP_rc += 1
            elif test_rootcause_labels[idx_test].dtype.metadata['label'] == 'batt_sim':
                if test_rootcause_channels[idx_test] in batt_sim_groundtruth_rootcause_channels:
                    TP_rc += 1
                else:
                    FP_rc += 1

        print(model_name)
        print(data_split)
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
        print('Precision:')
        print(TP / (TP + FP))
        print('Recall:')
        print(TP / (TP + FN))
        print('F1 score:')
        print(TP / (TP + 0.5 * (FP + FN)))
        print('Area under precision-recall curve:')
        print(a_pr)
        print('Average detection delay in seconds:')
        print(np.mean(total_delays))
        print('Rootcause precision:')
        print(TP_rc / (FP_rc + TP_rc))

        tf.keras.backend.clear_session()
