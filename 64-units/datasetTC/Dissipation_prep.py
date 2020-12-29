"""
Load, process and clean the data for the Dissipation dataset

"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from datasetTC import DATASET_DIR
from sklearn.utils import shuffle


class DissipationDataset:

    def __init__(self):

        path = DATASET_DIR / 'D_features_case.xlsx'
        self.X_train = self._load_dataset(path)
        self.X_train = self.X_train.drop(['v', 'w',
                                          'drho_x', 'drho_y', 'drho_z',
                                          'drhou_x', 'drhou_y', 'drhou_z',
                                          'drhov_x', 'drhov_y', 'drhov_z',
                                          'drhow_x', 'drhow_y', 'drhow_z'], axis=1)
        path1 = DATASET_DIR / 'DL_label_case.xlsx'
        self.y = self._load_dataset(path1)
        self.y_train = self.y.drop(['y/d'], axis=1)

        path = DATASET_DIR / '395_D_features_case.xlsx'
        self.X_test = self._load_dataset(path)
        self.X_test = self.X_test.drop(['v', 'w',
                                        'drho_x', 'drho_y', 'drho_z',
                                        'drhou_x', 'drhou_y', 'drhou_z',
                                        'drhov_x', 'drhov_y', 'drhov_z',
                                        'drhow_x', 'drhow_y', 'drhow_z'], axis=1)

        path1 = DATASET_DIR / '395_D_label_case.xlsx'
        self.y = self._load_dataset(path1)
        self.y_test = self.y.drop(['y/d'], axis=1)

        # Split the dataset into train and test
        self.X_train_shuffled, self.y_train_shuffled\
            = shuffle(self.X_train, self.y_train, random_state=42)

        # testing function
        """
        df = self.X_test
        self.means = df.mean()
        self.stds = df.std()
        """

    def _load_dataset(self, path) -> pd.DataFrame:
        print(f'Loading following dataset: {path}.')
        df = pd.read_excel(path)
        return df

    def get_data(self, test: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns a deep copy of a DataFrame containing the columns
        that are continuous features, and another deep copy containing
        the label columns.

        """
        df_x = self.X_test if test else self.X_train_shuffled
        df_y = self.y_test if test else self.y_train_shuffled

        return df_x.copy(deep=True), df_y.copy(deep=True)

    def _set_normalization_values(self, test):
        # we do not standardize the test labels
        df = self.get_data(test)[0]
        self.min = df.min()
        self.max = df.max()

        df = (df - self.min) / (self.max - self.min)
        # df = df
        return df

    def get_data_validation(self, test: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Similar to get_data(), this function will take the non-shuffled data instead.
        """
        df_x = self.X_test if test else self.X_train
        df_y = self.y_test if test else self.y_train

        return df_x.copy(deep=True), df_y.copy(deep=True)

    def _set_normalization_values_validation(self, test):
        # we do not standardize the test labels
        df = self.get_data_validation(test)[0]
        self.min = df.min()
        self.max = df.max()

        df = (df - self.min) / (self.max - self.min)
        # df = df
        return df
