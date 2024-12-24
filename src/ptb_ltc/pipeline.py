from typing import List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler

from ptb_ltc.processing.features import (
    clean_categorical_vars,
)
from ptb_ltc.config.core import config

# Numeric features
# Impute missing values with mean
# Scaling
num_pipeline = Pipeline(
    [
        ("num_imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()), 
    ]
)

# Categorical features
# Impute missing values with mode
# One hot encode
cat_pipeline = Pipeline(
    [
        ("clean_cat_vars", clean_categorical_vars()),
        ("cat_imputer", SimpleImputer(strategy="most_frequent", missing_values=None)),
        ("ohe", OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
    ]
)


# Transform pipeline for model training
def get_transform_pipeline(
    numeric_features: List[str], categorical_features: List[str]
) -> object:
    transform_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return transform_pipeline


# Get RandomUnderSampler
def get_undersample_transformer()->object:
  return RandomUnderSampler(sampling_strategy=config.processing.RANDOM_UNDER_SAMPLING_STRATEGY, 
                                         random_state=config.general.RANDOM_STATE)