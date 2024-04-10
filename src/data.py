"""
Lucas Correia
Mercedes-Benz AG
Mercedesstr. 137 | 70327 Stuttgart | Germany
"""

import os
import pickle
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import utility_functions
from tensorflow.keras.layers import *

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Load pickled data
data_load_path = 'Path to data'
data_save_path = 'Path to data'

# Load raw data
# -> normal_list: list of normal sequences
# -> anomalous_list: list of anomalous sequences, each numpy array within has metadata indicating its type of anomaly
normal_list = utility_functions.load_pickle(os.path.join(data_load_path, 'normal.pkl'))
anomalous_list = utility_functions.load_pickle(os.path.join(data_load_path, 'anomalous.pkl'))

# Iterate through normal_list and find the time in hours each sequence represents
meas_time = []
for normal_ts in normal_list:
    meas_time.append(len(sequence) / (2*60*60))  # 2*60*60 converts samples to hours

# Find total dynamic testing time
cumsum = np.cumsum(meas_time)

# Find index of sequence that comes closest to 1h of testing time
_, idx_1 = utility_functions.find_nearest(cumsum, 1)
# Find index of sequence that comes closest to 8h of testing time
_, idx_8 = utility_functions.find_nearest(cumsum, 8)
# Find index of sequence that comes closest to 64h of testing time
_, idx_64 = utility_functions.find_nearest(cumsum, 64)
# Find index of sequence that comes closest to 512h of testing time
_, idx_512 = utility_functions.find_nearest(cumsum, 512)

train_list_1h = normal_list[:idx_1]
train_list_8h = normal_list[:idx_8]
train_list_64h = normal_list[:idx_64]
train_list_512h = normal_list[:idx_512]
train_list_list = [train_list_1h, train_list_8h, train_list_64h, train_list_512h]
test_list_normal = normal_list[idx_512:]
test_list = test_list_normal + anomalous_list
random.shuffle(test_list)

data_split_list = ['1h', '8h', '64h', '512h']

# Find the most suitable window size (power of two) and the corresponding window shift
window_size = max([ts_processor.find_window_size(series) for series in train_list_resampled])
window_size_corrected = 2 ** (math.ceil(math.log(window_size, 2)))
window_shift = window_size_corrected // 2

for data_splits_idx, train_list in enumerate(train_list_list):
    train_list, val_list = train_test_split(train_list, random_state=seed, test_size=0.2)

    # Find mean, std, etc. metrics
    scalers = utility_functions.find_scalers(train_list)

    # Scale data
    scaled_train_list = list(utility_functions.scale_list(train_list, scalers, 'z-score'))
    scaled_val_list = list(utility_functions.scale_list(val_list, scalers, 'z-score'))

    # Window list
    scaled_train_window = utility_functions.window_list(scaled_train_list, window_size, window_shift)
    scaled_val_window = utility_functions.window_list(scaled_val_list, window_size, window_shift)

    # Create tf.data objects
    tfdata_train = tf.data.Dataset.from_tensor_slices(scaled_train_window.astype(np.dtype('float32')))
    tfdata_val = tf.data.Dataset.from_tensor_slices(scaled_val_window.astype(np.dtype('float32')))

    # Shuffle and batch tf.data objects
    tfdata_train = tfdata_train.shuffle(tfdata_train.cardinality(), seed=seed).batch(1).prefetch(tf.data.AUTOTUNE)
    tfdata_val = tfdata_val.shuffle(tfdata_val.cardinality(), seed=seed).batch(1).prefetch(tf.data.AUTOTUNE)

    # Save training and validation data
    tf.data.Dataset.save(tfdata_train, os.path.join(data_save_path, data_split_list[data_splits_idx], 'train'))
    tf.data.Dataset.save(tfdata_val, os.path.join(data_save_path, data_split_list[data_splits_idx], 'val'))

    # scale test data
    scaled_test_list = list(utility_functions.scale_list(test_list, scalers, 1))

    # Save training data as pickle files
    new_file = 'train.pkl'
    utility_functions.dump_pickle(os.path.join(data_save_path, data_split_list[data_splits_idx], new_file), scaled_train_list)

    # Save validation data as pickle files
    new_file = 'val.pkl'
    utility_functions.dump_pickle(os.path.join(data_save_path, data_split_list[data_splits_idx], new_file), scaled_val_list)

    # Save test data as pickle files
    new_file = 'test.pkl'
    utility_functions.dump_pickle(os.path.join(data_save_path, data_split_list[data_splits_idx], new_file), scaled_test_list)
