"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import pickle
import numpy as np


class BaseProcessor:
    @staticmethod
    def load_pickle(
            file_dir,
    ):
        """
        This method loads a pickle file.

        :param file_dir: directory to file to load
        """

        with open(file_dir, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def dump_pickle(
            obj,
            file_dir,
    ):
        """
        This method dumps an object to a pickle file.

        :param obj: object to dump
        :param file_dir: directory to file to load
        """

        with open(file_dir, 'wb') as file:
            pickle.dump(obj, file)
        return

    @staticmethod
    def window_array(
            input_array: np.ndarray,
            window_size: int = None,
            window_shift: int = None,
    ) -> np.ndarray:
        """
        Generates windows from an array.

        :param input_array: multivariate time series of shape (number_of_timesteps, channels)
        :param window_size: window size
        :param window_shift: window shift
        :return: array of windows of shape (number_of_windows, window_size, channels)
        """

        assert window_size is not None, "window_size must be provided."
        assert window_shift is not None, "window_shift must be provided."

        time_steps, channels = input_array.shape
        number_of_windows = (time_steps - window_size) // window_shift + 1
        output_windows = np.lib.stride_tricks.as_strided(
            input_array,
            shape=(number_of_windows, window_size, channels),
            strides=(window_shift * input_array.strides[0], input_array.strides[0], input_array.strides[1])
        )
        return output_windows
