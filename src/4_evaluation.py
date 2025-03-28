"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os
import numpy as np
import pickle
import random
import tensorflow as tf
import sklearn.metrics
from scipy import integrate
from utilities import detection_class
from dotenv import dotenv_values

# Declare constants
MODEL_NAME = 'tevae'  # or 'omnianomaly', 'sisvae', 'lwvae', 'vsvae', 'vasp', 'wvae', 'noma'
reverse_window_mode = 'mean'  # 'last', 'first'

# Load variables in .env file
config = dotenv_values("../.env")

# Load directory paths from .env file
data_path = config['data_path']
model_path = config['model_path']

data_split_list = ['1h', '8h', '64h', '512h']

channel_list = [
    'Vehicle Speed [-]',
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

motor_pump_middle_groundtruth_rootcause_channels = [10, 11, 12]
motor_pump_beginning_groundtruth_rootcause_channels = [10, 11, 12]
wheel_diameter_groundtruth_rootcause_channels = [0]
recup_off_groundtruth_rootcause_channels = [1, 2, 3, 9]
batt_sim_groundtruth_rootcause_channels = [6, 7, 8, 9]

for model_seed in range(1, 4):
    for data_split in data_split_list:
        # Declare model name and paths
        model_name = MODEL_NAME + '_' + data_split + '_' + str(model_seed)
        data_load_path = os.path.join(data_path, '2_preprocessed')
        model_load_path = os.path.join(model_path, model_name)

        # Load tf.data to get window_size
        tfdata_train = tf.data.Dataset.load(os.path.join(data_load_path, data_split, 'train'))

        detector = detection_class.AnomalyDetector(
            model_path=model_load_path,
            window_size=tfdata_train.element_spec.shape[0],
            sampling_rate=2,
            original_sampling_rate=10,
            calculate_delay=True,
            label_keyword='normal',
        )

        # Load data
        val_list = detector.load_pickle(os.path.join(data_load_path, data_split, 'val.pkl'))
        test_list = detector.load_pickle(os.path.join(data_load_path, data_split, 'test.pkl'))

        # Load inference results for validation data
        val_detection_score_list = detector.load_pickle(os.path.join(model_load_path, 'val_detection_score.pkl'))
        val_output = detector.load_pickle(os.path.join(model_load_path, 'val_output.pkl'))

        # Load inference results for test data
        test_detection_score_list = detector.load_pickle(os.path.join(model_load_path, 'test_detection_score.pkl'))
        test_output = detector.load_pickle(os.path.join(model_load_path, 'test_output.pkl'))

        # Evaluate the model
        threshold = detector.unsupervised_threshold(val_detection_score_list)

        groundtruth_labels, predicted_labels, total_delays = detector.evaluate_online(
            input_list=test_list,
            detection_score_list=test_detection_score_list,
            threshold=threshold,
        )

        # Evaluate test data using unsupervised threshold  # TODO
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
                        delay, _ = ts_processor.find_detection_delay(score_ts, threshold, sampling_rate, reverse_window_mode, window_size, len(score_ts), groundtruth_anomaly_start)
                        total_delays.append(abs(delay))
                        test_rootcause_channels.append(np.nan)
                        test_rootcause_labels.append(label)
                # =0 time steps in anomaly score higher than threshold
                # False negative
                else:
                    FN += 1
                    delay = (len(score_ts) - groundtruth_anomaly_start) / sampling_rate
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
                        delay, _ = ts_processor.find_detection_delay(score_ts, threshold, sampling_rate, reverse_window_mode, window_size, len(score_ts), groundtruth_anomaly_start)
                        total_delays.append(abs(delay))
                        test_rootcause_channels.append(np.nan)
                        test_rootcause_labels.append(label)
                # =0 time steps in anomaly score higher than threshold
                # False negative
                else:
                    FN += 1
                    delay = (len(score_ts) - groundtruth_anomaly_start) / sampling_rate
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
