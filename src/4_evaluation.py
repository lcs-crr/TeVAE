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

results = []
results_best = []
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

        groundtruth_labels, predicted_labels, total_delays, predicted_rootcause = detector.evaluate_online(
            input_list=test_list,
            detection_score_list=test_detection_score_list,
            threshold=threshold,
        )

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

        results.append({
            'Seed': model_seed,
            'Fold': fold_idx,
            'F1': metrics.f1_score(groundtruth_labels, predicted_labels, zero_division=0.0),
            'Precision': metrics.precision_score(groundtruth_labels, predicted_labels, zero_division=0.0),
            'Recall': metrics.recall_score(groundtruth_labels, predicted_labels, zero_division=0.0),
            'Delay': np.mean(total_delays),
            'Rootcause Precision': TP_rc / (TP_rc + FP_rc) if TP_rc + FP_rc > 0 else 0.0,
            'Threshold': threshold
        })

        f1_list = []
        reduced_test_detection_score = np.concatenate(test_detection_score_list).ravel()
        percentile_array = np.arange(0, 100.01, 0.01)
        for threshold_percentile in percentile_array:
            threshold_temp = np.percentile(reduced_test_detection_score, threshold_percentile)
            groundtruth_labels_temp, predicted_labels_temp, _, _ = detector.evaluate_online(
                input_list=test_list,
                detection_score_list=test_detection_score_list,
                threshold=threshold_temp,
            )
            f1_list.append(metrics.f1_score(groundtruth_labels_temp, predicted_labels_temp, zero_division=0.0))
        f1_list = np.vstack(f1_list)
        a_pr = metrics.auc(recall_list[:, 0], precision_list[:, 0])
        threshold_best = np.percentile(reduced_test_detection_score, percentile_array[np.argmax(f1_list)]).astype(float)

        groundtruth_labels_best, predicted_labels_best, total_delays_best, predicted_rootcause = detector.evaluate_online(
            input_list=test_list,
            detection_score_list=test_detection_score_list,
            threshold=threshold_best,
        )

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

        results_best.append({
            'Seed': model_seed,
            'Fold': fold_idx,
            'F1': metrics.f1_score(groundtruth_labels_best, predicted_labels_best, zero_division=0.0),
            'Precision': metrics.precision_score(groundtruth_labels_best, predicted_labels_best, zero_division=0.0),
            'Recall': metrics.recall_score(groundtruth_labels_best, predicted_labels_best, zero_division=0.0),
            'Delay': np.mean(total_delays_best),
            'Rootcause Precision': TP_rc/(TP_rc+FP_rc) if TP_rc+FP_rc > 0 else 0.0,
            'Threshold': threshold_best
        })

results = pd.DataFrame(results)
results_best = pd.DataFrame(results_best)

if not os.path.isfile(os.path.join(model_path, 'results.xlsx')):
    # Create and save a valid Excel file
    wb = openpyxl.Workbook()
    wb.save(os.path.join(model_path, 'results.xlsx'))

# Use a try-finally block to ensure proper handling
try:
    with pd.ExcelWriter(os.path.join(model_path, 'results.xlsx'), mode='a', if_sheet_exists='overlay') as writer:
        results.to_excel(writer, index=False, sheet_name=MODEL_NAME + '_' + AD_MODE)
        results_best.to_excel(writer, index=False, sheet_name=MODEL_NAME + '_' + AD_MODE + '_best')
finally:
    # Cleanup: Remove default 'Sheet' if it exists
    try:
        workbook = openpyxl.load_workbook(os.path.join(model_path, 'results.xlsx'))
        if 'Sheet' in workbook.sheetnames:
            del workbook['Sheet']
        workbook.save(os.path.join(model_path, 'results.xlsx'))
    except Exception as e:
        print(f"Error cleaning up sheets: {e}")
