"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os
import pickle
import random
import math
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from utilities import data_class
from dotenv import dotenv_values

# Declare constants
SEED = 1

# Set fixed seed for random operations
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Load variables in .env file
config = dotenv_values("../.env")

# Load directory paths from .env file
data_path = config['data_path']
model_path = config['model_path']

data_processor = data_class.DataProcessor(
    scale_method='z-score',
    sampling_rate=2,
)

data_load_path = os.path.join(data_path, '1_parsed')
data_save_path = os.path.join(data_path, '2_preprocessed')

# Load raw data
normal_list = data_processor.load_pickle(os.path.join(data_load_path, 'normal.pkl'))
anomalous_list = data_processor.load_pickle(os.path.join(data_load_path, 'anomalous.pkl'))

split_idcs = data_processor.split_into_hours(normal_list, [1, 8, 64, 512])
split_names = ['1h', '8h', '64h', '512h']

for split_idx, split in enumerate(split_idcs):
    train_list = normal_list[:split]
    test_list_normal = normal_list[split_idcs[-1]:]
    test_list = test_list_normal + anomalous_list
    random.shuffle(test_list)

    # Split training data into training and validation
    train_list, val_list = train_test_split(train_list, random_state=SEED, test_size=0.2)

    # Find the scalers for each feature
    data_processor.find_scalers_from_list(train_list)

    # Scale data
    train_list_scaled = data_processor.scale_list(train_list)
    val_list_scaled = data_processor.scale_list(val_list)
    test_list_scaled = data_processor.scale_list(test_list)

    # Find the window size
    data_processor.find_window_size_from_list(train_list_scaled)

    # Window sequences inside lists
    scaled_train_window = data_processor.window_list(train_list_scaled)
    scaled_val_window = data_processor.window_list(val_list_scaled)

    # Create tf.data objects
    tfdata_train = tf.data.Dataset.from_tensor_slices(scaled_train_window)
    tfdata_val = tf.data.Dataset.from_tensor_slices(scaled_val_window)

    # Shuffle and batch tf.data objects
    tfdata_train = tfdata_train.shuffle(tfdata_train.cardinality(), seed=SEED)
    tfdata_val = tfdata_val.shuffle(tfdata_val.cardinality(), seed=SEED)

    # Save training and validation data as tf.data
    tf.data.Dataset.save(tfdata_train, os.path.join(data_save_path, split_names[split_idx], 'train'))
    tf.data.Dataset.save(tfdata_val, os.path.join(data_save_path, split_names[split_idx], 'val'))

    # Save training, validation and testing data as pickle files
    data_processor.dump_pickle(train_list_scaled, os.path.join(data_save_path, split_names[split_idx], 'train.pkl'))
    data_processor.dump_pickle(val_list_scaled, os.path.join(data_save_path, split_names[split_idx], 'val.pkl'))
    data_processor.dump_pickle(test_list_scaled, os.path.join(data_save_path, split_names[split_idx], 'test.pkl'))
