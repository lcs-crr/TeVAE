"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os
from typing import Optional, List, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from model_garden import tevae, sisvae, omnianomaly, lwvae, vsvae, vasp, wvae, noma
from utilities import base_class


class Inferencer(base_class.BaseProcessor):
    def __init__(
            self,
            model_path: str = None,
            window_size: int = None,
            window_shift: int = None,
            reverse_mode: str = 'mean',
            batch_size: int = None,
    ) -> None:
        """
        This class comprises all required functions to perform inference with a given model.

        :param model_path: path to the trained model
        :param window_size: window size
        :param window_shift: window shift
        :param reverse_mode: reverse window mode
        :param batch_size: batch size
        """
        super().__init__()
        self.model_path = model_path
        self.window_size = window_size
        self.window_shift = window_shift
        self.reverse_mode = reverse_mode
        self.batch_size = batch_size

    def _reverse_window_array(
            self,
            input_windows: np.ndarray,
    ) -> np.ndarray:
        """
        Generic wrapper for methods that generates multivariate time series from windows.

        :param input_windows: array of windows of shape (number_of_windows, window_size, channels)
        :return: multivariate time series of shape (number_of_timesteps, channels)
        """

        assert isinstance(input_windows, np.ndarray), 'input_windows must be a numpy array!'
        assert input_windows.ndim == 3, 'input_windows must be a 3D numpy array of shape (num_windows, window_size, channels)!'
        assert self.reverse_mode is not None, 'reverse_mode must be provided!'
        assert self.window_shift is not None, 'window_shift must be provided!'

        if self.reverse_mode == 'mean':
            return self._mean_reverse_window_array(input_windows)
        elif self.reverse_mode == 'last':
            return self._last_reverse_window_array(input_windows)
        elif self.reverse_mode == 'first':
            return self._first_reverse_window_array(input_windows)
        else:
            raise ValueError('Invalid reverse-window method!')

    def _mean_reverse_window_array(
            self,
            input_windows: np.ndarray,
    ) -> np.ndarray:
        """
        Private function that generates a multivariate time series from windows.

        :param input_windows: array of windows of shape (number_of_windows, window_size, channels)
        :return: multivariate time series of shape (number_of_timesteps, channels)
        """

        assert isinstance(input_windows, np.ndarray), 'input_windows must be a numpy array!'
        assert input_windows.ndim == 3, 'input_windows must be a 3D numpy array of shape (num_windows, window_size, channels)!'
        assert self.window_shift is not None, 'window_shift must be provided!'

        num_windows, window_size, num_channels = input_windows.shape
        total_length = (num_windows - 1) * self.window_shift + window_size
        # Initialize arrays for summing values and counting contributions
        data_sum = np.zeros((total_length, num_channels))
        data_count = np.zeros((total_length, num_channels))
        # Populate data_sum and data_count with overlapping windows
        for window_index in range(num_windows):
            start_idx = window_index * self.window_shift
            end_idx = start_idx + window_size
            data_sum[start_idx:end_idx] += input_windows[window_index]
            data_count[start_idx:end_idx] += 1
        return data_sum / data_count

    def _first_reverse_window_array(
            self,
            input_windows: np.ndarray,
    ) -> np.ndarray:
        """
        Private function that generates a multivariate time series from windows.

        :param input_windows: array of windows of shape (number_of_windows, window_size, channels)
        :return: multivariate time series of shape (number_of_timesteps, channels)
        """

        assert isinstance(input_windows, np.ndarray), 'input_windows must be a numpy array!'
        assert input_windows.ndim == 3, 'input_windows must be a 3D numpy array of shape (num_windows, window_size, channels)!'
        assert self.window_shift is not None, 'window_shift must be provided!'

        num_windows, window_size, num_channels = input_windows.shape
        output_size = (num_windows - 1) * self.window_shift + window_size  # Compute final output size
        # Pre-allocate output array
        data = np.zeros((output_size, num_channels))
        # Use vectorized assignment
        indices = np.arange(num_windows - 1) * self.window_shift  # Compute start indices for each window
        data[indices[:, None] + np.arange(self.window_shift), :] = input_windows[:-1, :self.window_shift]
        # Assign the last window
        data[-window_size:, :] = input_windows[-1]
        return data

    def _last_reverse_window_array(
            self,
            input_windows: np.ndarray,
    ) -> np.ndarray:
        """
        Private function that generates a multivariate time series from windows.

        :param input_windows: array of windows of shape (number_of_windows, window_size, channels)
        :return: multivariate time series of shape (number_of_timesteps, channels)
        """

        assert isinstance(input_windows, np.ndarray), 'input_windows must be a numpy array!'
        assert input_windows.ndim == 3, 'input_windows must be a 3D numpy array of shape (num_windows, window_size, channels)!'
        assert self.window_shift is not None, 'window_shift must be provided!'

        num_windows, window_size, num_channels = input_windows.shape
        output_size = (num_windows - 1) * self.window_shift + window_size  # Compute final output size
        # Pre-allocate output array
        data = np.zeros((output_size, num_channels))
        # Assign the first window
        data[:window_size, :] = input_windows[0]
        # Compute the start indices for each assignment
        indices = (np.arange(1, num_windows) - 1) * self.window_shift + window_size
        data[indices[:, None] + np.arange(self.window_shift), :] = input_windows[1:, -self.window_shift:]
        return data

    def _get_model_name_from_path(
            self,
    ) -> str:
        """
        Extract model name from model directory.

        :return: model name
        """

        assert self.model_path is not None, 'model_path must be provided!'

        return self.model_path.split('/')[-1][:-5]

    def _load_keras_model(
            self,
            model_name: str,
    ) -> tf.keras.Model:
        """
        Private function to load keras model.

        :param model_name: name of model
        :return: Keras model
        """

        assert self.model_path is not None, 'model_path must be provided!'

        if model_name == 'tevae':
            model = tf.keras.models.load_model(self.model_path, custom_objects={
                'TEVAE_Encoder': tevae.TEVAE_Encoder,
                'TEVAE_Decoder': tevae.TEVAE_Decoder,
                'MA': tevae.MA,
                'TEVAE': tevae.TEVAE,
            })
        elif model_name == 'sisvae':
            model = tf.keras.models.load_model(self.model_path, custom_objects={
                'SISVAE_Encoder': sisvae.SISVAE_Encoder,
                'SISVAE_Decoder': sisvae.SISVAE_Decoder,
                'SISVAE': sisvae.SISVAE,
            })
        elif model_name == 'omnianomaly':
            model = tf.keras.models.load_model(self.model_path, custom_objects={
                'OmniAnomaly': omnianomaly.OmniAnomaly,
                'OmniAnomaly_Encoder': omnianomaly.OmniAnomaly_Encoder,
                'OmniAnomaly_Decoder': omnianomaly.OmniAnomaly_Decoder,
            })
        elif model_name == 'lwvae':
            model = tf.keras.models.load_model(self.model_path, custom_objects={
                'LWVAE': lwvae.LWVAE,
                'LWVAE_Encoder': lwvae.LWVAE_Encoder,
                'LWVAE_Decoder': lwvae.LWVAE_Decoder,
            })
        elif model_name == 'vsvae':
            model = tf.keras.models.load_model(self.model_path, custom_objects={
                'VSVAE': vsvae.VSVAE,
                'VSVAE_Encoder': vsvae.VSVAE_Encoder,
                'VSVAE_Decoder': vsvae.VSVAE_Decoder,
                'VS': vsvae.VS,
            })
        elif model_name == 'vasp':
            model = tf.keras.models.load_model(self.model_path, custom_objects={
                'VASP': vasp.VASP,
                'VASP_Encoder': vasp.VASP_Encoder,
                'VASP_Decoder': vasp.VASP_Decoder,
            })
        elif model_name == 'wvae':
            model = tf.keras.models.load_model(self.model_path, custom_objects={
                'WVAE': wvae.WVAE,
                'WVAE_Encoder': wvae.WVAE_Encoder,
                'WVAE_Decoder': wvae.WVAE_Decoder,
            })
        elif model_name == 'noma':
            model = tf.keras.models.load_model(self.model_path, custom_objects={
                'NOMA': noma.NOMA,
                'NOMA_Encoder': noma.NOMA_Encoder,
                'NOMA_Decoder': noma.NOMA_Decoder,
            })
        else:
            raise ValueError('Model name not found!')
        return model

    def inference_list(
            self,
            input_list: List[np.ndarray],
            subset_name: str,
            save_inference_results: Optional[bool] = True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[np.ndarray]]]:
        """
        Inference function.

        :param input_list: list of multivariate time series, each of shape (number_of_timesteps, channels)
        :param subset_name: subset name, such as 'train', 'val', 'test'
        :param save_inference_results: flag to save inference results
        :return: list of detection scores, each of shape (number_of_timesteps, channels)
        :return: list of model outputs (mean, std, sampled), each of shape (number_of_timesteps, channels)
        """

        assert isinstance(input_list, list), 'input_list argument must be a list!'
        assert all(isinstance(input_array, np.ndarray) for input_array in input_list), 'All items in input_list must be numpy arrays!'
        assert all(input_array.ndim == 2 for input_array in input_list), 'All items in input_list must be 2D numpy arrays!'
        assert self.model_path is not None, 'model_path must be provided!'

        _inference_array = {
            'tevae': self._tevae_inference,
            'sisvae': self._sisvae_inference,
            'omnianomaly': self._omnianomaly_inference,
            'lwvae': self._lwvae_inference,
            'vsvae': self._vsvae_inference,
            'vasp': self._vasp_inference,
            'wvae': self._wvae_inference,
            'noma': self._noma_inference,
        }

        model_name = self._get_model_name_from_path()
        model = self._load_keras_model(model_name)

        detection_score_list = []
        rootcause_score_list = []
        output_list = []
        for data_ts in tqdm(input_list):
            detection_score, rootcause_score, output = _inference_array[model_name](model, data_ts)
            detection_score_list.append(detection_score)
            rootcause_score_list.append(rootcause_score)
            output_list.append(output)
        if save_inference_results:
            self.dump_pickle(detection_score_list, os.path.join(self.model_path, subset_name + '_detection_score' + '.pkl'))
            self.dump_pickle(rootcause_score_list, os.path.join(self.model_path, subset_name + '_rootcause_score' + '.pkl'))
            self.dump_pickle(output_list, os.path.join(self.model_path, subset_name + '_output' + '.pkl'))
        return detection_score_list, rootcause_score_list, output_list

    def _tevae_inference(
            self,
            model: tf.keras.Model,
            input_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Private inference function for TeVAE.

        :param input_array: multivariate time series of shape (number_of_timesteps, channels)
        :return: detection score
        :return: list of model outputs (mean, std, sample)
        """

        assert input_array.ndim == 2, 'input_array must be a 2D numpy array (time_steps, channels)!'
        assert self.window_size is not None, 'window_size must be provided!'
        assert self.window_shift is not None, 'window_shift must be provided!'

        # Window input array
        input_windows = self.window_array(input_array, self.window_size, self.window_shift)
        # Predict output windows
        xhat_mean, xhat_logvar, xhat, _, _, _, _ = model.predict(input_windows, batch_size=self.batch_size, verbose=0)
        # Calculate variance parameter
        xhat_var = np.exp(xhat_logvar)
        # Reverse window mean parameter
        xhat_mean = self._reverse_window_array(xhat_mean)
        # Reverse window variance parameter
        xhat_var = self._reverse_window_array(xhat_var)
        # Reverse window reconstruction
        xhat = self._reverse_window_array(xhat)
        # Calculate standard deviation parameter
        xhat_std = np.sqrt(xhat_var)
        # Calculate anomaly score
        anomaly_score = model.rec_fn(input_array, [xhat_mean, np.log(xhat_var)], reduce_time=False).numpy()
        # Calculate rootcause scores
        rootcause_score = -tfp.distributions.Normal(loc=xhat_mean, scale=xhat_std).log_prob(input_array)
        # Clear GPU memory before next call
        tf.keras.backend.clear_session()
        # Return anomaly score and model outputs
        return anomaly_score, rootcause_score, [xhat_mean, xhat_std, xhat]

    def _sisvae_inference(
            self,
            model: tf.keras.Model,
            input_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Private inference function for SISVAE.

        :param model: trained model
        :param input_array: multivariate time series of shape (number_of_timesteps, channels)
        :return: detection score
        :return: list of model outputs (mean, std, sample)
        """

        assert input_array.ndim == 2, 'input_array must be a 2D numpy array (time_steps, channels)!'
        assert self.window_size is not None, 'window_size must be provided!'
        assert self.window_shift is not None, 'window_shift must be provided!'

        # Window input array
        input_windows = self.window_array(input_array, self.window_size, self.window_shift)
        # Predict output windows
        xhat_mean, xhat_logvar, xhat, _, _, _ = model.predict(input_windows, batch_size=self.batch_size, verbose=0)
        # Calculate variance parameter
        xhat_var = np.exp(xhat_logvar)
        # Reverse window mean parameter
        xhat_mean = self._reverse_window_array(xhat_mean)
        # Reverse window variance parameter
        xhat_var = self._reverse_window_array(xhat_var)
        # Reverse window reconstruction
        xhat = self._reverse_window_array(xhat)
        # Calculate standard deviation parameter
        xhat_std = np.sqrt(xhat_var)
        # Calculate anomaly score
        anomaly_score = model.rec_fn(input_array, [xhat_mean, np.log(xhat_var)], reduce_time=False).numpy()
        # Calculate rootcause scores
        rootcause_score = -tfp.distributions.Normal(loc=xhat_mean, scale=xhat_std).log_prob(input_array)
        # Clear GPU memory before next call
        tf.keras.backend.clear_session()
        # Return anomaly score and model outputs
        return anomaly_score, rootcause_score, [xhat_mean, xhat_std, xhat]

    def _omnianomaly_inference(
            self,
            model: tf.keras.Model,
            input_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Private inference function for OmniAnomaly.

        :param model: trained model
        :param input_array: multivariate time series of shape (number_of_timesteps, channels)
        :return: detection score
        :return: list of model outputs (mean, std, sample)
        """

        assert input_array.ndim == 2, 'input_array must be a 2D numpy array (time_steps, channels)!'
        assert self.window_size is not None, 'window_size must be provided!'
        assert self.window_shift is not None, 'window_shift must be provided!'

        # Window input array
        input_windows = self.window_array(input_array, self.window_size, self.window_shift)
        # Predict output windows
        xhat_mean, xhat_logvar, xhat, _, _, _ = model.predict(input_windows, batch_size=self.batch_size, verbose=0)
        # Calculate variance parameter
        xhat_var = np.exp(xhat_logvar)
        # Reverse window mean parameter
        xhat_mean = self._reverse_window_array(xhat_mean)
        # Reverse window variance parameter
        xhat_var = self._reverse_window_array(xhat_var)
        # Reverse window reconstruction
        xhat = self._reverse_window_array(xhat)
        # Calculate standard deviation parameter
        xhat_std = np.sqrt(xhat_var)
        # Calculate anomaly score
        anomaly_score = model.rec_fn(input_array, [xhat_mean, np.log(xhat_var)], reduce_time=False).numpy()
        # Calculate rootcause scores
        rootcause_score = -tfp.distributions.Normal(loc=xhat_mean, scale=xhat_std).log_prob(input_array)
        # Clear GPU memory before next call
        tf.keras.backend.clear_session()
        # Return anomaly score and model outputs
        return anomaly_score, rootcause_score, [xhat_mean, xhat_std, xhat]

    def _lwvae_inference(
            self,
            model: tf.keras.Model,
            input_array: np.ndarray,
    ) -> Tuple[np.ndarray, None, np.ndarray]:
        """
        Private inference function for LW-VAE.

        :param model: trained model
        :param input_array: multivariate time series of shape (number_of_timesteps, channels)
        :return: detection score
        :return: list of model outputs (mean, std, sample)
        """

        assert input_array.ndim == 2, 'input_array must be a 2D numpy array (time_steps, channels)!'
        assert self.window_size is not None, 'window_size must be provided!'
        assert self.window_shift is not None, 'window_shift must be provided!'

        # Window input array
        input_windows = self.window_array(input_array, self.window_size, self.window_shift)
        # Predict output windows
        xhat, _, _, _, = model.predict(input_windows, batch_size=self.batch_size, verbose=0)
        # Reverse window reconstruction
        xhat = self._reverse_window_array(xhat)
        # Calculate anomaly score
        anomaly_score = model.rec_fn(input_array, xhat, reduce_time=False).numpy()
        # Clear GPU memory before next call
        tf.keras.backend.clear_session()
        # Return anomaly score and model outputs
        return anomaly_score, None, xhat

    def _vsvae_inference(
            self,
            model: tf.keras.Model,
            input_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Private inference function for VS-VAE.

        :param input_array: multivariate time series of shape (number_of_timesteps, channels)
        :return: detection score
        :return: list of model outputs (mean, std, sample)
        """

        assert input_array.ndim == 2, 'input_array must be a 2D numpy array (time_steps, channels)!'
        assert self.window_size is not None, 'window_size must be provided!'
        assert self.window_shift is not None, 'window_shift must be provided!'

        # Window input array
        input_windows = self.window_array(input_array, self.window_size, self.window_shift)
        # Predict output windows
        xhat_mean, xhat_logvar, xhat, _, _, _, _ = model.predict(input_windows, batch_size=self.batch_size, verbose=0)
        # Calculate variance parameter
        xhat_var = np.exp(xhat_logvar)
        # Reverse window mean parameter
        xhat_mean = self._reverse_window_array(xhat_mean)
        # Reverse window variance parameter
        xhat_var = self._reverse_window_array(xhat_var)
        # Reverse window reconstruction
        xhat = self._reverse_window_array(xhat)
        # Calculate standard deviation parameter
        xhat_std = np.sqrt(xhat_var)
        # Calculate anomaly score
        anomaly_score = model.rec_fn(input_array, [xhat_mean, np.log(xhat_var)], reduce_time=False).numpy()
        # Calculate rootcause scores
        rootcause_score = -tfp.distributions.Laplace(loc=xhat_mean, scale=xhat_std).log_prob(input_array)
        # Clear GPU memory before next call
        tf.keras.backend.clear_session()
        # Return anomaly score and model outputs
        return anomaly_score, rootcause_score, [xhat_mean, xhat_std, xhat]

    def _vasp_inference(
            self,
            model: tf.keras.Model,
            input_array: np.ndarray,
    ) -> Tuple[np.ndarray, None, np.ndarray]:
        """
        Private inference function for VASP.

        :param model: trained model
        :param input_array: multivariate time series of shape (number_of_timesteps, channels)
        :return: detection score
        :return: list of model outputs (mean, std, sample)
        """

        assert input_array.ndim == 2, 'input_array must be a 2D numpy array (time_steps, channels)!'
        assert self.window_size is not None, 'window_size must be provided!'
        assert self.window_shift is not None, 'window_shift must be provided!'

        # Window input array
        input_windows = self.window_array(input_array, self.window_size, self.window_shift)
        # Predict output windows
        xhat, _, _, _, = model.predict(input_windows, batch_size=self.batch_size, verbose=0)
        # Reverse window reconstruction
        xhat = self._reverse_window_array(xhat)
        # Calculate anomaly score
        anomaly_score = model.rec_fn(input_array, xhat, reduce_time=False).numpy()
        # Clear GPU memory before next call
        tf.keras.backend.clear_session()
        # Return anomaly score and model outputs
        return anomaly_score, None, xhat

    def _wvae_inference(
            self,
            model: tf.keras.Model,
            input_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Private inference function for WVAE.

        :param model: trained model
        :param input_array: multivariate time series of shape (number_of_timesteps, channels)
        :return: detection score
        :return: list of model outputs (mean, std, sample)
        """

        assert input_array.ndim == 2, 'input_array must be a 2D numpy array (time_steps, channels)!'
        assert self.window_size is not None, 'window_size must be provided!'
        assert self.window_shift is not None, 'window_shift must be provided!'

        # Window input array
        input_windows = self.window_array(input_array, self.window_size, self.window_shift)
        # Predict output windows
        xhat_mean, xhat_logvar, xhat, _, _, _ = model.predict(input_windows, batch_size=self.batch_size, verbose=0)
        # Calculate variance parameter
        xhat_var = np.exp(xhat_logvar)
        # Reverse window mean parameter
        xhat_mean = self._reverse_window_array(xhat_mean)
        # Reverse window variance parameter
        xhat_var = self._reverse_window_array(xhat_var)
        # Reverse window reconstruction
        xhat = self._reverse_window_array(xhat)
        # Calculate standard deviation parameter
        xhat_std = np.sqrt(xhat_var)
        # Calculate anomaly score
        anomaly_score = model.rec_fn(input_array, [xhat_mean, np.log(xhat_var)], reduce_time=False).numpy()
        # Calculate rootcause scores
        rootcause_score = -tfp.distributions.Normal(loc=xhat_mean, scale=xhat_std).log_prob(input_array)
        # Clear GPU memory before next call
        tf.keras.backend.clear_session()
        # Return anomaly score and model outputs
        return anomaly_score, rootcause_score, [xhat_mean, xhat_std, xhat]

    def _noma_inference(
            self,
            model: tf.keras.Model,
            input_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Private inference function for NOMA.

        :param model: trained model
        :param input_array: multivariate time series of shape (number_of_timesteps, channels)
        :return: detection score
        :return: list of model outputs (mean, std, sample)
        """

        assert input_array.ndim == 2, 'input_array must be a 2D numpy array (time_steps, channels)!'
        assert self.window_size is not None, 'window_size must be provided!'
        assert self.window_shift is not None, 'window_shift must be provided!'

        # Window input array
        input_windows = self.window_array(input_array, self.window_size, self.window_shift)
        # Predict output windows
        xhat_mean, xhat_logvar, xhat, _, _, _ = model.predict(input_windows, batch_size=self.batch_size, verbose=0)
        # Calculate variance parameter
        xhat_var = np.exp(xhat_logvar)
        # Reverse window mean parameter
        xhat_mean = self._reverse_window_array(xhat_mean)
        # Reverse window variance parameter
        xhat_var = self._reverse_window_array(xhat_var)
        # Reverse window reconstruction
        xhat = self._reverse_window_array(xhat)
        # Calculate standard deviation parameter
        xhat_std = np.sqrt(xhat_var)
        # Calculate anomaly score
        anomaly_score = model.rec_fn(input_array, [xhat_mean, np.log(xhat_var)], reduce_time=False).numpy()
        # Calculate rootcause scores
        rootcause_score = -tfp.distributions.Normal(loc=xhat_mean, scale=xhat_std).log_prob(input_array)
        # Clear GPU memory before next call
        tf.keras.backend.clear_session()
        # Return anomaly score and model outputs
        return anomaly_score, rootcause_score, [xhat_mean, xhat_std, xhat]