# Custom data processing steps, need to be compliant with sklearn pipeline
from typing import List

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ptb_ltc.config.core import config


# This processing step clean all the values of categorical columns by replacing special characters with underscores
class clean_categorical_vars(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Customerized class for cleaning all the values of categorical columns by replacing special characters with underscores"""
        pass

    def fit(
        self, X, y=None
    ):  # need to have y as argument to make class compatible with sklearn pipeline
        """Fit

        Args:
            X (DataFrame): a input dataframe of features to train the transformer
            y (DataFrame): a input Series of response variable to train the transformer (optional)
        """
        return self

    def transform(self, X):
        """Transform

        Args:
            X (DataFrame): a input dataframe of features to be transformed

        Returns:
            X (DataFrame): the transformed Dataframe of features
        """
        X = X.copy()

        for col in X.columns:
            X[col] = (
                X[col]
                .fillna(
                    "MissingValuePlaceHolder"
                )  # fill missing values with a placeholder string
                .astype(str)
                .str.strip()
                .str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
                .replace(
                    "MissingValuePlaceHolder", None
                )  # replace placeholder string with None
                .str.lower()
            )

        return X

    def get_feature_names_out(self):
        pass