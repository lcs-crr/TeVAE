"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import math
import numpy as np
from scipy.signal import butter, lfilter, sosfilt
from statsmodels.tsa import stattools
from typing import List, Union
from utilities import base_class


class DataProcessor(base_class.BaseProcessor):
    def __init__(
            self,
            window_size: int = None,
            original_sampling_rate: int = None,
            target_sampling_rate: int = None,
            scale_method: str = None,
            window_shift: Union[int, str] = 'half',
    ) -> None:
        """
        This class comprises all required functions to process the data.

        :param window_size: window size
        :param original_sampling_rate: original sampling rate of raw data
        :param target_sampling_rate: target sampling rate of processed data
        :param scale_method: scaling method, either 'z-score' or 'min-max'
        :param window_shift: window overlap, integer refers to number of time steps, half to half a window
        """
        super().__init__()
        self.window_size = window_size
        self.original_sampling_rate = original_sampling_rate
        self.target_sampling_rate = target_sampling_rate
        self.scale_method = scale_method
        self.window_shift = window_shift
        self.minimum = None
        self.maximum = None
        self.mean = None
        self.standard_deviation = None

    def find_window_size_from_list(
            self,
            data_list: List[np.ndarray],
    ) -> None:
        """
        This function finds the window size required to model the dynamics in all channels of a multivariate time series.

        :param data_list: list of multivariate input time series
        """

        # Pre-allocate list for possible window sizes
        possible_window_sizes = []
        # Iterate through list of multivariate time series
        for data_ts in data_list:
            intersection_list = []
            # Iterate through channels
            for channel in range(data_ts.shape[-1]):
                corr_array = stattools.acf(data_ts[:, channel], alpha=0.01, nlags=4096)
                upper_y = corr_array[1][:, 1] - corr_array[0]
                corr = corr_array[0]
                try:
                    intersection_list.append(np.min(np.where(corr - upper_y < 0)[0]))
                except:
                    continue
            # Append maximum window size for each channel
            possible_window_sizes.append(np.max(intersection_list))
        # Choose maximum window size in list
        candidate_window_size = np.max(possible_window_sizes)
        # If window_size is not supposed to be rounded, return it as is
        round_to_power = 2
        upper_window_size = round_to_power ** (np.ceil(math.log(candidate_window_size, round_to_power)))
        lower_window_size = round_to_power ** (np.floor(math.log(candidate_window_size, round_to_power)))
        # Find which rounded window size is closer to the candidate window size and return it
        if upper_window_size - candidate_window_size > candidate_window_size - lower_window_size:
            self.window_size = int(lower_window_size)
        else:
            self.window_size = int(upper_window_size)

    def window_list(
            self,
            input_list: List[np.ndarray],
    ) -> np.ndarray:
        """
        Generates windows from an array.

        :param input_list: multivariate time series of shape (number_of_timesteps, channels)
        :return: array of windows of shape (number_of_windows, window_size, channels)
        """

        assert isinstance(input_list, list), 'input_list argument must be a list!'
        assert all(isinstance(input_array, np.ndarray) for input_array in input_list), 'All items in input_list must be numpy arrays!'
        assert all(input_array.ndim == 2 for input_array in input_list), 'All items in input_list must be 2D numpy arrays!'
        assert self.window_size is not None, 'To window the sequences in the list, find the window size first by running the find_window_size_from_list method first!'

        if not isinstance(self.window_shift, int):
            if self.window_shift == 'half':
                self.window_shift = self.window_size // 2

        output_list = [self.window_array(input_array, self.window_size, self.window_shift) for input_array in input_list]
        return np.vstack(output_list)

    def find_scalers_from_list(
            self,
            input_list: List[np.ndarray],
    ) -> None:
        """
        This function finds the channel-wise minimum, maximum, mean and standard deviation for each channel in a list of arrays.

        :param input_list: list of multivariate time series, each of shape (number_of_timesteps, channels)
        """

        self.minimum = np.min(np.vstack(input_list), axis=0)
        self.maximum = np.max(np.vstack(input_list), axis=0)
        self.mean = np.mean(np.vstack(input_list), axis=0)
        self.standard_deviation = np.std(np.vstack(input_list), axis=0)

    def scale_list(
            self,
            input_list: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        This function scales a list of multivariate time series according to the provided method.

        :param input_list: list of multivariate time series, each of shape (number_of_timesteps, channels)
        """

        assert isinstance(input_list, list), 'input_list argument must be a list!'
        assert all(isinstance(input_array, np.ndarray) for input_array in input_list), 'All items in input_list must be numpy arrays!'
        assert all(input_array.ndim == 2 for input_array in input_list), 'All items in input_list must be 2D numpy arrays!'
        assert self.minimum is not None, 'To scale the list, find the scalers first by running the find_scalers_from_list method first!'
        assert self.maximum is not None, 'To scale the list, find the scalers first by running the find_scalers_from_list method first!'
        assert self.mean is not None, 'To scale the list, find the scalers first by running the find_scalers_from_list method first!'
        assert self.standard_deviation is not None, 'To scale the list, find the scalers first by running the find_scalers_from_list method first!'

        if self.scale_method == 'z-score':
            return [(time_series - self.mean) / self.standard_deviation for time_series in input_list]
        elif self.scale_method == 'min-max':
            return [(time_series - self.minimum) / (self.maximum - self.minimum) for time_series in input_list]
        elif self.scale_method is None:
            return [time_series for time_series in input_list]
        else:
            raise ValueError('Invalid scaling method!')

    def split_into_hours(
            self,
            input_list: List[np.ndarray],
            hour_splits: List[int],
    ) -> List[int]:
        """
        This method splits input_list into chunks as long as the hours specified in hour_splits.

        :param input_list: list of multivariate time series, each of shape (number_of_timesteps, channels)
        :param hour_splits: list of split sizes in hours
        :return:
        """

        if self.downsampled_flag:
            sampling_rate = self.target_sampling_rate
        else:
            sampling_rate = self.original_sampling_rate

        time_duration = []
        for _, input_array in enumerate(input_list):
            time_duration.append(len(input_array) / (sampling_rate * 60 * 60))  # 2*60*60 to convert to hours
        cum_time_duration = np.cumsum(time_duration)

        splits_idcs = []
        for idx_hour in hour_splits:
            closest_idx = self.find_nearest_index(cum_time_duration, idx_hour)
            splits_idcs.append(closest_idx)
        return splits_idcs

    @staticmethod
    def find_nearest_index(
            input_array: np.ndarray,
            target_value: float,
    ):
        """
        This method finds the index of the value inside input_array which is closest to target_value.

        :param input_array: array of cumulative time durations of a data set
        :param target_value: value to search closest neighbour to
        """

        array = np.asarray(input_array)
        closest_idx = (np.abs(array - target_value)).argmin()
        return closest_idx