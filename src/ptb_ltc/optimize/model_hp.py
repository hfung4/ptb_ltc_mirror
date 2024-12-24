from typing import Dict, Union

import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import Apply, scope


def _knn_clf_hp(n_neighbors: Union[int, Apply] = None) -> Dict[str, Apply]:
    """_knn_clf_hp
    Args:
        n_neighbors: Number of neighbors to use, default is None

    Returns:
        param_space (Dict): hyperparmeter search space for the classifier
    """
    param_space = dict(
        model__n_neighbors=(
            scope.int(hp.uniform("n_neighbors", 1, 15))
            if n_neighbors is None
            else n_neighbors
        )
    )
    return param_space


def _decision_tree_clf_regr_hp(
    criterion: Union[str, Apply] = None,
    max_depth: Union[int, Apply] = None,
    max_features: Union[str, int, float, Apply] = None,
    min_samples_split: Union[int, float, Apply] = None,
) -> Dict[str, Apply]:
    """_decision_tree_clf_regr_hp

    Args:
        criterion (Union[str, Apply], optional): The function to measure the quality of a split. Defaults to None.
        max_depth (Union[int, Apply], optional): The maximum depth of the tree. Defaults to None.
        max_features (Union[str, int, float, Apply], optional): The number of features to consider when looking for the best split. Defaults to None.
        min_samples_split (Union[int, float, Apply], optional): The minimum number of samples required to split an internal node. Defaults to None.

    Returns:
        param_space (Dict): hyperparmeter search space for the regressor or classifier
    """
    param_space = dict(
        model__criterion=(
            hp.choice("dtree_criterion", ["gini", "entropy"])
            if criterion is None
            else criterion
        ),
        model__max_depth=(
            scope.int(hp.qlognormal("dtree_max_depth", 3, 1, 1))
            if max_depth is None
            else max_depth
        ),
        model__max_features=(
            hp.choice("dtree_max_features", ["sqrt", "log2"])
            if max_features is None
            else max_features
        ),
        model__min_samples_split=(
            hp.choice("dtree_min_samples_split", [2, 5, 10, 50, 100, 300, 500, 1000])
            if min_samples_split is None
            else min_samples_split
        ),
    )
    return param_space


def _logistic_regression_hp(
    C: Union[float, Apply] = None,
    solver: Union[str, Apply] = None,
    class_weight: Union[str, Apply] = None,
) -> Dict[str, Apply]:
    """logistic_regression_hp_space

    Args:
        C (Union[float, Apply], optional): Inverse of regularization strength. Defaults to None.
        solver (Union[str, Apply], optional): Algorithm to use in the optimization problem. Defaults to None.
        class_weight (Union[str, Apply], optional): Weights associated with classes. Defaults to None.

    Returns:
       param_space (Dict): hyperparmeter search space for the regressor
    """
    param_space = dict(
        model__C=hp.uniform("logreg__C", 0.1, 2) if C is None else C,
        model__solver=(
            hp.choice("logreg_solver", ["liblinear", "lbfgs"])
            if solver is None
            else solver
        ),
        model__class_weight=(
            hp.choice("logreg_class_weight", ["balanced", None])
            if class_weight is None
            else class_weight
        ),
    )
    return param_space


def _random_forest_clf_regr_hp(
    criterion: Union[str, Apply] = None,
    class_weight: Union[str, Apply] = None,
    n_estimators: Union[int, Apply] = None,
    max_depth: Union[int, Apply] = None,
    min_samples_split: Union[float, Apply] = None,
    min_samples_leaf: Union[float, Apply] = None,
    max_features: Union[str, float, Apply] = None,
):
    """_random_forest_clf_regr_hp

    Args:
        criterion (Union[str, Apply], optional): The function to measure the quality of a split. Defaults to None.
        class_weight (Union[dict, list, Apply], optional): Weights associated with classes. Defaults to None.
        n_estimators (Union[int, Apply], optional): The number of trees in the forest. Defaults to None.
        max_depth (Union[int, Apply], optional): The maximum depth of the tree. Defaults to None.
        min_samples_split (Union[float, Apply], optional): The minimum number of samples required to split an internal node. Defaults to None.
        min_samples_leaf (Union[float, Apply], optional): The minimum number of samples required to be at a leaf node. Defaults to None.
        max_features (Union[str, float, Apply], optional): The number of features to consider when looking for the best split. Defaults to None.

    Returns:
        param_space (Dict): hyperparmeter search space for the regressor or classifier
    """
    param_space = dict(
        model__criterion=(
            hp.choice("rf_criterion", ["gini", "entropy"])
            if criterion is None
            else criterion
        ),
        model__class_weight=(
            hp.choice("rf_class_weight", ["balanced", "balanced_subsample", None])
            if class_weight is None
            else class_weight
        ),
        model__n_estimators=(
            scope.int(hp.qloguniform("rf_n_estimators", np.log(9.5), np.log(1200.5), 1))
            if n_estimators is None
            else n_estimators
        ),
        model__max_depth=(
            hp.pchoice(
                "rf_max_depth",
                [
                    (0.7, None),  # most common choice.
                    (0.1, 2),  # try some shallow trees.
                    (0.1, 3),
                    (0.1, 4),
                ],
            )
            if max_depth is None
            else max_depth
        ),
        model__min_samples_split=(
            hp.pchoice(
                "rf_min_samples_split",
                [
                    (0.95, 2),  # most common choice
                    (0.05, 3),  # try minimal increase
                ],
            )
            if min_samples_split is None
            else min_samples_split
        ),
        model__min_samples_leaf=(
            hp.choice(
                "rf_min_samples_leaf",
                [
                    1,  # most common choice.
                    scope.int(
                        hp.qloguniform(
                            "rf_min_samples_leaf_gt1",
                            np.log(1.5),
                            np.log(50.5),
                            1,
                        )
                    ),
                ],
            )
            if min_samples_leaf is None
            else min_samples_leaf
        ),
        model__max_features=(
            hp.pchoice(
                "rf_max_features",
                [
                    (0.2, "sqrt"),  # most common choice.
                    (0.1, "log2"),  # less common choice.
                    (0.1, None),  # all features, less common choice.
                    (0.6, hp.uniform("rf_max_features_frac", 0.0, 1.0)),
                ],
            )
            if max_features is None
            else max_features
        ),
    )
    return param_space


def _xgb_clf_regr_hp(
    max_depth: Union[int, Apply] = None,
    learning_rate: Union[float, Apply] = None,
    n_estimators: Union[int, Apply] = None,
    gamma: Union[float, Apply] = None,
    min_child_weight: Union[float, Apply] = None,
    subsample: Union[float, Apply] = None,
    colsample_bytree: Union[float, Apply] = None,
    colsample_bylevel: Union[float, Apply] = None,
    reg_alpha: Union[float, Apply] = None,
    reg_lambda: Union[float, Apply] = None,
):
    """_xgb_clf_regr_hp

    Args:
        max_depth (Union[int, Apply], optional): Maximum tree depth for base learners. Defaults to None.
        learning_rate (Union[float, Apply], optional): Boosting learning rate. Defaults to None.
        n_estimators (Union[int, Apply], optional): Number of boosted trees to fit. Defaults to None.
        gamma (Union[float, Apply], optional): Regularisation parameter. Defaults to None.
        min_child_weight (Union[float, Apply], optional): Minimum sum of instance weight (hessian) needed in a child. Defaults to None.
        subsample (Union[float, Apply], optional): Subsample ratio of the training instances. Defaults to None.
        colsample_bytree (Union[float, Apply], optional): Subsampling parameter for columns. Defaults to None.
        colsample_bylevel (Union[float, Apply], optional): Subsampling parameter for columns. Defaults to None.
        reg_alpha (Union[float, Apply], optional): L1 regularization term. Defaults to None.
        reg_lambda (Union[float, Apply], optional): L2 regularization term. Defaults to None.

    Returns:
        param_space (Dict): hyperparmeter search space for the regressor or classifier
    """
    param_space = dict(
        model__max_depth=(
            scope.int(hp.uniform("xgb_max_depth", 1, 11))
            if max_depth is None
            else max_depth
        ),
        model__learning_rate=(
            (hp.loguniform("xgb_learning_rate", np.log(0.0001), np.log(0.5)) - 0.0001)
            if learning_rate is None
            else learning_rate
        ),
        model__n_estimators=(
            scope.int(hp.quniform("xgb_n_estimators", 100, 2000, 200))
            if n_estimators is None
            else n_estimators
        ),
        model__gamma=(
            (hp.loguniform("xgb_gamma", np.log(0.0001), np.log(5)) - 0.0001)
            if gamma is None
            else gamma
        ),
        model__min_child_weight=(
            hp.loguniform("xgb_min_child_weight", np.log(1), np.log(100))
            if min_child_weight is None
            else min_child_weight
        ),
        model__subsample=(
            hp.uniform("xgb_subsample", 0.5, 1) if subsample is None else subsample
        ),
        model__colsample_bytree=(
            hp.uniform("xgb_colsample_bytree", 0.5, 1)
            if colsample_bytree is None
            else colsample_bytree
        ),
        model__colsample_bylevel=(
            hp.uniform("xgb_colsample_bylevel", 0.5, 1)
            if colsample_bylevel is None
            else colsample_bylevel
        ),
        model__reg_alpha=(
            (hp.loguniform("xgb_reg_alpha", np.log(0.0001), np.log(1)) - 0.0001)
            if reg_alpha is None
            else reg_alpha
        ),
        model__reg_lambda=(
            hp.loguniform("xgb_reg_lambda", np.log(1), np.log(4))
            if reg_lambda is None
            else reg_lambda
        ),
    )
    return param_space


def _lgbm_clf_regr_hp(
    max_depth: Union[int, Apply] = None,
    num_leaves: Union[int, Apply] = None,
    learning_rate: Union[float, Apply] = None,
    n_estimators: Union[int, Apply] = None,
    min_child_weight: Union[float, Apply] = None,
    subsample: Union[float, Apply] = None,
    colsample_bytree: Union[float, Apply] = None,
    reg_alpha: Union[float, Apply] = None,
    reg_lambda: Union[float, Apply] = None,
    boosting_type: Union[str, Apply] = None,
):
    """_lgbm_clf_regr_hp

    Args:
        max_depth (Union[int, Apply], optional): Maximum tree depth for base learners. Defaults to None.
        num_leaves (Union[int, Apply], optional): Maximum tree leaves for base learners. Defaults to None.
        learning_rate (Union[float, Apply], optional): Boosting learning rate. Defaults to None.
        n_estimators (Union[int, Apply], optional): Number of boosted trees to fit. Defaults to None.
        min_child_weight (Union[float, Apply], optional):  Minimum sum of instance weight (Hessian) needed in a child (leaf). Defaults to None.
        subsample (Union[float, Apply], optional): Subsample ratio of the training instance. Defaults to None.
        colsample_bytree (Union[float, Apply], optional): Subsample ratio of columns when constructing each tree. Defaults to None.
        reg_alpha (Union[float, Apply], optional): L1 regularization term. Defaults to None.
        reg_lambda (Union[float, Apply], optional): L2 regularization term. Defaults to None.
        boosting_type (Union[str, Apply], optional): Boosting type. Defaults to None.

    Returns:
        param_space (Dict): hyperparmeter search space for the regressor or classifier
    """
    param_space = dict(
        model__max_depth=(
            scope.int(hp.uniform("lgbm_max_depth", 1, 11))
            if max_depth is None
            else max_depth
        ),
        model__num_leaves=(
            scope.int(hp.uniform("lgbm_num_leaves", 2, 121))
            if num_leaves is None
            else num_leaves
        ),
        model__learning_rate=(
            (hp.loguniform("lgbm_learning_rate", np.log(0.0001), np.log(0.5)) - 0.0001)
            if learning_rate is None
            else learning_rate
        ),
        model__n_estimators=(
            scope.int(hp.quniform("lgbm_n_estimators", 100, 2000, 200))
            if n_estimators is None
            else n_estimators
        ),
        model__min_child_weight=(
            scope.int(hp.loguniform("lgbm_min_child_weight", np.log(1), np.log(100)))
            if min_child_weight is None
            else min_child_weight
        ),
        model__subsample=(
            hp.uniform("lgbm_subsample", 0.5, 1) if subsample is None else subsample
        ),
        model__colsample_bytree=(
            hp.uniform("lgbm_colsample_bytree", 0.5, 1)
            if colsample_bytree is None
            else colsample_bytree
        ),
        model__reg_alpha=(
            (hp.loguniform("lgbm_reg_alpha", np.log(0.0001), np.log(1)) - 0.0001)
            if reg_alpha is None
            else reg_alpha
        ),
        model__reg_lambda=(
            hp.loguniform("lgbm_reg_lambda", np.log(1), np.log(4))
            if reg_lambda is None
            else reg_lambda
        ),
        model__boosting_type=(
            hp.choice("lgbm_boosting_type", ["gbdt", "dart", "goss"])
            if boosting_type is None
            else boosting_type
        ),
    )
    return param_space


def _mlp_clf_hp(
    activation: Union[str, Apply] = None,
    solver: Union[str, Apply] = None,
    alpha: Union[float, Apply] = None,
    learning_rate: Union[str, Apply] = None,
    power_t: Union[float, Apply] = None,
    max_iter: Union[int, Apply] = None,
    tol: Union[float, Apply] = None,
    momentum: Union[float, Apply] = None,
):
    """_mlp_clf_hp

    Args:
        activation (Union[str, Apply], optional): Activation function for the hidden layer. Defaults to None.
        solver (Union[str, Apply], optional): The solver for weight optimization. Defaults to None.
        alpha (Union[float, Apply], optional): Strength of the L2 regularization term.. Defaults to None.
        learning_rate (Union[str, Apply], optional): The initial learning rate used. Defaults to None.
        power_t (Union[float, Apply], optional): The exponent for inverse scaling learning rate. Defaults to None.
        max_iter (Union[int, Apply], optional): Maximum number of iterations.. Defaults to None.
        tol (Union[float, Apply], optional): Tolerance for the optimization. Defaults to None.
        momentum (Union[float, Apply], optional): Momentum for gradient descent update. Defaults to None.

    Returns:
        param_space (Dict): hyperparmeter search space for the classifier
    """
    param_space = dict(
        model__activation=(
            hp.pchoice(
                "mlp_activation",
                [
                    (0.2, "identity"),
                    (0.2, "logistic"),
                    (0.2, "tanh"),
                    (0.4, "relu"),
                ],
            )
            if activation is None
            else activation
        ),
        model__solver=(
            hp.pchoice("mlp_solver", [(0.2, "lbfgs"), (0.2, "sgd"), (0.6, "adam")])
            if solver is None
            else solver
        ),
        model__alpha=hp.uniform("mlp_alpha", 1e-4, 0.01) if alpha is None else alpha,
        model__learning_rate=(
            hp.choice("mlp_learning_rate", ["constant", "invscaling", "adaptive"])
            if learning_rate is None
            else learning_rate
        ),
        model__power_t=(
            hp.uniform("mlp_power_t", 0.1, 0.9) if power_t is None else power_t
        ),
        model__max_iter=(
            scope.int(hp.uniform("mlp_max_iter", 150, 350))
            if max_iter is None
            else max_iter
        ),
        model__tol=hp.uniform("mlp_tol", 1e-4, 0.01) if tol is None else tol,
        model__momentum=(
            hp.uniform("mlp_momentum", 0.8, 1.0) if momentum is None else momentum
        ),
    )
    return param_space
