from hyperopt import hp

import ptb_ltc.optimize.model_hp as model_hp
import ptb_ltc.optimize.smote_hp as smote_hp
from ptb_ltc.config.core import config

# Specify the hyperparameters for each model ---------------------------------------------------

# KNN Classifier
knn_user_specified_hp = {}  # Can read from a json or yaml file
knn_param_space = model_hp._knn_clf_hp(**knn_user_specified_hp)

# Decision Tree Classifier
dtree_user_params = {}  # Can read from a json or yaml file
dtree_param_space = model_hp._decision_tree_clf_regr_hp(**dtree_user_params)

# Logistic Regression
logreg_user_params = {}  # Can read from a json or yaml file
logreg_param_space = model_hp._logistic_regression_hp(**logreg_user_params)

# Random Forest Classifier
rf_user_params = {}  # Can read from a json or yaml file
random_forest_param_space = model_hp._random_forest_clf_regr_hp(**rf_user_params)

# XGBoost Classifier
xgb_user_params = {}  # Can read from a json or yaml file
xgb_param_space = model_hp._xgb_clf_regr_hp(**xgb_user_params)

# LightGBM Classifier
lgbm_user_params = {}  # Can read from a json or yaml file
lgbm_param_space = model_hp._lgbm_clf_regr_hp(**lgbm_user_params)

# Multi-layer Perceptron Classifier
mlp_user_params = {}  # Can read from a json or yaml file
mlp_param_space = model_hp._mlp_clf_hp(**mlp_user_params)


# Define a full list of param_space, one for each available model type
full_model_param_space = [
    # KNN
    {"type": "KNeighborsClassifier", "params": knn_param_space},
    # Naive Bayes
    {"type": "GaussianNB", "params": {}},
    # Decision Tree
    {"type": "DecisionTreeClassifier", "params": dtree_param_space},
    # Logistic Regression
    {"type": "LogisticRegression", "params": logreg_param_space},
    # RandomForestClassifier
    {"type": "RandomForestClassifier", "params": random_forest_param_space},
    # XGBClassifier
    {"type": "XGBClassifier", "params": xgb_param_space},
    # lightgbm
    {"type": "LGBMClassifier", "params": lgbm_param_space},
    # Neural Network: multi-layer perceptron
    {"type": "MLPClassifier", "params": mlp_param_space},
]

# Get a subset of the full_model_param_space based on user input (from model config)
selected_model_param_space = [
    d for d in full_model_param_space if d["type"] in config.model.MODELS_TO_TRY
]

# Final model parameter space, as a dictionary of hp.choice objects
final_model_param_space = dict(classifier=hp.choice("classifiers", selected_model_param_space))



# Specify the hyperparameters for each types of SMOTE -----------------------------------------------------

# SMOTE
smote_user_params = {}  # Can read from a json or yaml file
smote_param_space = smote_hp._smote_hp(**smote_user_params)

# Borderline SMOTE
borderline_smote_user_params = {}  # Can read from a json or yaml file
borderline_smote_param_space = smote_hp._borderline_smote_hp(
    **borderline_smote_user_params
)

# ADASYN
adasyn_user_params = {}  # Can read from a json or yaml file
adasyn_param_space = smote_hp._adasyn_hp(**adasyn_user_params)


# Define a full list of sampling param_space for SMOTE, one for each available sampling type
full_sampling_param_space = [
    {"type": "SMOTE", "params": smote_param_space},
    {"type": "BorderlineSMOTE", "params": borderline_smote_param_space},
    {"type": "ADASYN", "params": adasyn_param_space},
]

# Get a subset of the full_sampling_param_space for SMOTE based on user input (from sampling config)
selected_sampling_param_space = [
    d for d in full_sampling_param_space if d["type"] in config.processing.SMOTES_TO_TRY
]

# Final sampling parameter space for SMOTE, as a dictionary of hp.choice objects
final_sampling_param_space = dict(
    sampler=hp.choice("samplers", selected_sampling_param_space)
)

# Combine the model, and sampling parameter space if ENABLE_RESAMPLING is enabled
if config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE:
    param_space = {
        **final_sampling_param_space,
        **final_model_param_space,
    }
else:
    param_space = final_model_param_space


if __name__ == "__main__":
    # testing
    print(param_space)
