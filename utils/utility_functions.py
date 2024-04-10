# Lucas Correia
# PhD Candidate - Online Multivariate Time-series Anomaly Detection
# RD/EPDT - Evaluation Systems & Calibration Methods
# Mercedes-Benz AG | Factory 019
# Mercedesstraße 137 | HPC D650 | 70546 Stuttgart | Germany

import numpy as np
from scipy.signal import butter, lfilter
import tensorflow_probability as tfp
import tensorflow as tf
import scipy
from statsmodels.tsa import stattools
import os
import pickle
from scipy.interpolate import interp1d
import sklearn


def generator_whole(data):
    """
    This function generates a generator for whole sequences.

    :param data: list of multivariate time series
    :type data: list[array (time steps, channels)]
    :return: whatever the output is
    """
    for i in range(len(data)):
        yield data[i][:, :], data[i][:, :]


def generator_window(data):
    """
    This function generates a generator for windows.

    :param data: windows of multivariate time series
    :type data: array (number of windows, window size, channels)
    :return: whatever the output is
    """
    for i in range(len(data)):
        yield data[i, :, :], data[i, :, :]


def window_list(input_list, window_size, shift):
    """
    This function generates windows from a list of arrays.
    The windows are concatenated into a single array.

    :param input_list: list of multivariate time series
    :type input_list: list[array (time steps, channels)]
    :param window_size: window size
    :type window_size: int
    :param shift: number of time steps between windows
    :type shift: int
    :return: array of windows
    :rtype: array (number of windows, window size, channels)
    """
    # If input is a single array, convert it to a list
    if isinstance(input_list, np.ndarray):
        input_list = [input_list]
    # Pre-allocate list
    window_list_temp = []
    # For each multivariate time series in the list
    for time_series in input_list:
        # Calculate the number of windows
        set_window_count = (time_series.shape[0] - window_size) // shift + 1
        # Pre-allocate array
        window_data = np.zeros((set_window_count, window_size, time_series.shape[1]))
        # For each window
        for j in range(set_window_count):
            window_data[j] = time_series[j * shift:window_size + j * shift]
        # Append windows to list
        window_list_temp.append(window_data)
    # Concatenate windows into a single array
    windows = np.concatenate(window_list_temp[:], axis=0)
    # Return windows
    return windows


def reverse_window(windows, shift, mode):
    """
    This function reconstructs a continuous multivariate time series from windows.

    :param windows: array of windows
    :type windows: array (number of windows, window size, channels)
    :param shift: time steps between windows
    :type shift: int
    :param mode: reverse window mode
    :type mode: str
    :return: multivariate time series
    :rtype: array (time steps, channels)
    """
    num_windows, window_size, num_channels = windows.shape
    data = np.zeros(((num_windows - 1) * shift + window_size, num_channels))  # Pre-allocate array
    if mode == 'last':
        data[:window_size, :] = windows[0, :, :]  # First window
        for i in range(1, num_windows):
            data[(i - 1) * shift + window_size: i * shift + window_size, :] = windows[i, -shift:, :]
    elif mode == 'first':
        for i in range(num_windows - 1):
            data[i * shift: (i + 1) * shift, :] = windows[i, :shift, :]
        data[-window_size:, :] = windows[-1, :, :]  # Last window
    elif mode == 'mean':
        data = np.full((num_windows, (num_windows - 1) * shift + window_size, num_channels), np.nan)  # Pre-allocate array
        for i in range(num_windows):
            data[i, i * shift: i * shift + window_size] = windows[i]
        data = np.nanmean(data, axis=0)
    return data


def find_scalers(input_list):
    """
    This function finds the minimum, maximum, mean and standard deviation for each channel in a list of arrays.

    :param input_list: list of multivariate time series
    :type input_list: list[array (time steps, channels)]
    :return: list of scalers
    :rtype: list[array (channels)]
    """
    # Calculate minimum of all time series for each channel
    minimum = np.min(np.vstack(input_list), axis=0)
    # Calculate maximum of all time series for each channel
    maximum = np.max(np.vstack(input_list), axis=0)
    # Calculate mean of all time series for each channel
    mean = np.mean(np.vstack(input_list), axis=0)
    # Calculate standard deviation of all time series for each channel
    standard_deviation = np.std(np.vstack(input_list), axis=0)
    # Return scalers
    return [minimum, maximum, mean, standard_deviation]


def scale_list(input_list, scalers, scale_type):
    """
    This function scales a multivariate time series. The scalers input must come from find_scalers() function.
    :param input_list: list of multivariate time series
    :type input_list: list[array (time steps, channels)]
    :param scalers: list of scalers
    :type scalers: list[array (channels)]
    :param scale_type: scaling type
    :type scale_type: str
    :return:
    """
    # Assign minimum scaler value from scalers input
    minimum = scalers[0]
    # Assign maximum scaler value from scalers input
    maximum = scalers[1]
    # Assign mean scaler value from scalers input
    mean = scalers[2]
    # Assign standard deviation scaler value from scalers input
    standard_deviation = scalers[3]
    # If scale type is z-score
    if scale_type == 'z-score':
        data_scaled = [(time_series - mean) / standard_deviation for time_series in input_list]
    # Else if scale type is min-max
    elif scale_type == 'min-max':
        data_scaled = [(time_series - minimum) / (maximum - minimum) for time_series in input_list]
    # Else do nothing
    else:
        data_scaled = [time_series for time_series in input_list]
    # Return scaled data
    return data_scaled


def downsample(time_series, cutoff_frequency, sampling_frequency, filter_order):
    """
    This function applies a low-pass Butterworth filter to a multivariate time series.
    :param time_series: multivariate time series
    :type time_series: array (time steps, channels)
    :param cutoff_frequency: cutoff frequency
    :type cutoff_frequency: float
    :param sampling_frequency: sampling frequency
    :type sampling_frequency: float
    :param filter_order: filter order
    :type filter_order: int
    :return: low-pass filtered multivariate time series
    :rtype: array (time steps, channels)
    """
    # Calculate Butterworth filter coefficients
    b, a = butter(filter_order, cutoff_frequency, fs=sampling_frequency, btype='low', analog=False)
    # Apply filter to time series
    filtered_time_series = lfilter(b, a, time_series)
    # Return filtered time series
    return filtered_time_series


def upsample(time_series, freq_target):
    # Upsample to target frequency
    time_axis = np.round(time_series.timestamps - time_series.timestamps[0], 3)
    signal = time_series.samples
    f = interp1d(time_axis, signal)
    time_new = np.arange(0, time_axis[-1], 1 / freq_target)
    signal_new = f(time_new)
    return signal_new


def find_similarity(time_series1, time_series2, mode='pearson'):
    """
    This function calculates the similarity between two univariate time series.

    :param time_series1: first univariate time series
    :type time_series1: array (time steps)
    :param time_series2: second univariate time series
    :type time_series2: array (time steps)
    :param mode: similarity mode
    :type mode: str
    :return:
    """
    # If mode is Pearson correlation
    if mode == 'pearson':
        similarity = scipy.stats.pearsonr(time_series1, time_series2)[0]
    # If mode is Euclidean distance
    elif mode == 'euclidean':
        similarity = scipy.spatial.distance.euclidean(time_series1, time_series2)
    # If mode is cosine similarity
    elif mode == 'cosine_similarity':
        similarity = scipy.spatial.distance.cosine(time_series1, time_series2)
    # If mode is manhattan distance
    elif mode == 'manhattan':
        similarity = scipy.spatial.distance.cityblock(time_series1, time_series2)
    # If mode is correlation
    elif mode == 'correlation':
        similarity = scipy.spatial.distance.correlation(time_series1, time_series2)
    # If mode is squared euclidean distance
    elif mode == 'sqeuclidean':
        similarity = scipy.spatial.distance.sqeuclidean(time_series1.numpy(), time_series2.numpy())
    # If mode is coefficient of determination
    elif mode == 'r2':
        similarity = sklearn.metrics.r2_score(time_series1, time_series2)
    # If mode is mse
    elif mode == 'mse':
        similarity = sklearn.metrics.mean_squared_error(time_series1, time_series2)
    # If mode is component
    elif mode == 'component':
        idx = np.argmax(time_series1)
        similarity = time_series2[idx] / time_series1[idx]
    # If mode is max
    elif mode == 'max':
        idx = np.argmax(time_series1)
        similarity = time_series2[idx]
    # If mode is not provided
    else:
        similarity = 0
    # Return similarity
    return similarity


def negative_log_likelihood(mean, standard_deviation, sample):
    """
    Calculates the negative log likelihood for Gaussian distribution parameters given sample
    :param mean: Mean parameter of Gaussian distribution
    :type mean: array (M, features)
    :param standard_deviation: Standard deviation parameter of Gaussian distribution
    :type standard_deviation: array (M, features)
    :param sample: Observed sample
    :type sample: array (M, features)
    :return: negative log likelihood
    :rtype: array (M, features)
    """
    multi_outputDist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=standard_deviation)
    uni_outputDist = tfp.distributions.Normal(loc=mean, scale=standard_deviation)
    multi_negloglik = tf.expand_dims(-multi_outputDist.log_prob(sample), axis=1)  # (M, 1)
    uni_negloglik = -uni_outputDist.log_prob(sample)  # (M, features)
    return multi_negloglik, uni_negloglik


def inference_s_vae(model, input_array, rev_mode, window_size, batch_size=512):
    """
    Inference function for stochastic variational autoencoder.

    :param model: trained model
    :type model: tf.keras.Model
    :param input_array: multivariate time series
    :type input_array: array (time steps, channels)
    :param rev_mode: reverse window mode
    :type rev_mode: str
    :param window_size: window size
    :type window_size: int
    :param batch_size: batch size
    :type batch_size: int
    :return: multivariate and univariate negative log likelihood and model outputs
    :rtype: array (M, 1), array (M, features), list[list[array (M, features), array (M, features), array (M, features)], list[array (windows, window size, features), array (windows, window size, features), array (windows, window size, features)]]
    """
    # Window input array
    input_windows = window_list(input_array, window_size, 1)
    # Predict output windows
    output_windows = model.predict(input_windows,
                                   batch_size=batch_size,
                                   verbose=0,
                                   steps=None,
                                   callbacks=None)

    # Assign outputs
    Xhat_mean = output_windows[0]
    Xhat_logvar = output_windows[1]
    Xhat = output_windows[2]
    Z_mean = output_windows[3]
    Z_logvar = output_windows[4]
    Z = output_windows[5]
    # Calculate variance parameter
    Xhat_var = np.exp(Xhat_logvar)
    Z_var = np.exp(Z_logvar)
    # If window size is provided
    if window_size:
        # Reverse window mean parameter
        Xhat_mean = reverse_window(Xhat_mean, 1, rev_mode)
        # Reverse window variance parameter
        Xhat_var = reverse_window(Xhat_var, 1, rev_mode)
        # Reverse window reconstruction
        Xhat = reverse_window(Xhat, 1, rev_mode)
        try:
            # Reverse window mean parameter
            Z_mean = reverse_window(Z_mean, 1, rev_mode)
            # Reverse window variance parameter
            Z_var = reverse_window(Z_var, 1, rev_mode)
            # Reverse window reconstruction
            Z = reverse_window(Z, 1, rev_mode)
        except:
            pass
    # Calculate standard deviation parameter
    Xhat_std = np.sqrt(Xhat_var)
    Z_std = np.sqrt(Z_var)
    # Calculate multivariate and univariate negative log likelihood
    multi_negloglik, uni_negloglik = negative_log_likelihood(Xhat_mean, Xhat_std, input_array)
    # Clear GPU memory before next call
    tf.keras.backend.clear_session()
    # Return multivariate and univariate negative log likelihood and model outputs
    return multi_negloglik, uni_negloglik, [Xhat_mean, Xhat_std, Xhat], [Z_mean, Z_std, Z]


def inference_det_vae(model, input_array, rev_mode, window_size, batch_size=512):
    """
    Inference function for stochastic variational autoencoder.

    :param model: trained model
    :type model: tf.keras.Model
    :param input_array: multivariate time series
    :type input_array: array (time steps, channels)
    :param rev_mode: reverse window mode
    :type rev_mode: str
    :param window_size: window size
    :type window_size: int
    :param batch_size: batch size
    :type batch_size: int
    :return: multivariate and univariate negative log likelihood and model outputs
    :rtype: array (M, 1), array (M, features), list[list[array (M, features), array (M, features), array (M, features)], list[array (windows, window size, features), array (windows, window size, features), array (windows, window size, features)]]
    """
    # Window input array
    input_windows = window_list(input_array, window_size, 1)
    # Predict output windows
    output_windows = model.predict(input_windows,
                                   batch_size=batch_size,
                                   verbose=0,
                                   steps=None,
                                   callbacks=None)

    # Assign outputs
    Xhat = output_windows[0]
    Z_mean = output_windows[1]
    Z_logvar = output_windows[2]
    Z = output_windows[3]
    # Calculate variance parameter
    Z_var = np.exp(Z_logvar)
    # If window size is provided
    if window_size:
        # Reverse window reconstruction
        Xhat = reverse_window(Xhat, 1, rev_mode)
        try:
            # Reverse window mean parameter
            Z_mean = reverse_window(Z_mean, 1, rev_mode)
            # Reverse window variance parameter
            Z_var = reverse_window(Z_var, 1, rev_mode)
            # Reverse window reconstruction
            Z = reverse_window(Z, 1, rev_mode)
        except:
            pass
    # Calculate standard deviation parameter
    Z_std = np.sqrt(Z_var)
    # Calculate multivariate and univariate negative log likelihood
    # score = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(input_array, Xhat).numpy()[:, np.newaxis]

    score = (input_array - Xhat) ** 2
    score = tf.reduce_sum(score, axis=-1)[:, np.newaxis]

    # Clear GPU memory before next call
    tf.keras.backend.clear_session()
    # Return multivariate and univariate negative log likelihood and model outputs
    return score, np.repeat(score, Xhat.shape[-1], axis=-1), [Xhat, Xhat, Xhat], [Z_mean, Z_std, Z]


def find_window_size(series):
    """
    This function plots the autocorrelation for each channel in a multivariate time series.

    :param series: multivariate input time series
    :type series: array (time steps, channels)
    :return window size: integer
    """

    intersection_list = []
    for channel in range(series.shape[-1]):
        corr_array = stattools.acf(series[:, channel], alpha=0.01, nlags=4096)
        upper_y = corr_array[1][:, 1] - corr_array[0]
        # upper_y = np.around(upper_y, 3)
        corr = corr_array[0]
        # corr = np.around(corr, 3)
        try:
            intersection_list.append(min(np.where(corr - upper_y < 0)[0]))
        except:
            pass
    return np.max(intersection_list)


def find_detection_delay_old(score, threshold, sampling_frequency, rev_mode, window_size, sequence_length, anomaly_start):
    """
    This function calculates the total detection delay for a given reverse window mode.

    :param score: anomaly score
    :type score: array (M, 1)
    :param threshold: anomaly threshold
    :type threshold: float
    :param sampling_frequency: sampling frequency
    :type sampling_frequency: float
    :param rev_mode: reverse window mode
    :type rev_mode: str
    :param window_size: window size
    :type window_size: int
    :param sequence_length: sequence length
    :type sequence_length: int
    :param anomaly_start: anomaly start normalised by sequence_length
    :type anomaly_start: float
    :return: delay
    :rtype: float
    """
    # Find first time step above threshold
    time_step_detection = np.argwhere(score >= threshold)[0, 0]
    # If reverse window mode is mean or first
    if rev_mode == 'mean' or rev_mode == 'first':
        # If detection time step is before sequence_length - window_size
        if time_step_detection < sequence_length - window_size:
            rev_window_penalty = window_size
        # If detection time step is within last window_size time steps
        else:
            rev_window_penalty = sequence_length - time_step_detection
    elif rev_mode == 'last':
        # If detection time step is within first window_size time steps
        if time_step_detection < window_size:
            rev_window_penalty = window_size - time_step_detection
        # If detection time step is after first window_size time steps
        else:
            rev_window_penalty = 0
    # Sum detection delay with reverse window delay penalty and subtract in case of SS anomaly, then convert to seconds
    delay = (time_step_detection + rev_window_penalty - len(score) * anomaly_start) / sampling_frequency
    return delay, time_step_detection + rev_window_penalty


def find_detection_delay(score, threshold, sampling_frequency, rev_mode, window_size, sequence_length, anomaly_start):
    """
    This function calculates the total detection delay for a given reverse window mode.

    :param score: anomaly score
    :type score: array (M, 1)
    :param threshold: anomaly threshold
    :type threshold: float
    :param sampling_frequency: sampling frequency
    :type sampling_frequency: float
    :param rev_mode: reverse window mode
    :type rev_mode: str
    :param window_size: window size
    :type window_size: int
    :param sequence_length: sequence length
    :type sequence_length: int
    :param anomaly_start: time step of anomaly start
    :type anomaly_start: float
    :return: delay
    :rtype: float
    """
    # Find first time step above threshold
    time_step_detection = np.argwhere(score >= threshold)[0, 0]
    # If reverse window mode is mean or first
    if rev_mode == 'mean' or rev_mode == 'first':
        # If detection time step is before sequence_length - window_size
        if time_step_detection < sequence_length - window_size:
            rev_window_penalty = window_size
        # If detection time step is within last window_size time steps
        else:
            rev_window_penalty = sequence_length - time_step_detection
    elif rev_mode == 'last':
        # If detection time step is within first window_size time steps
        if time_step_detection < window_size:
            rev_window_penalty = window_size - time_step_detection
        # If detection time step is after first window_size time steps
        else:
            rev_window_penalty = 0
    # Sum detection delay with reverse window delay penalty and subtract in case of SS anomaly, then convert to seconds
    delay = (time_step_detection + rev_window_penalty - anomaly_start) / sampling_frequency
    return delay, time_step_detection + rev_window_penalty


def load_pickle(path):
    """
    :param path: path to load from
    :type path: str
    :return: pickle_obj: pickle object
    """
    with open(path, 'rb') as f:
        pickle_obj = pickle.load(f)
    return pickle_obj


def dump_pickle(path, obj):
    """
    :param path: path to load from
    :type path: str
    :param obj: object to dump
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
