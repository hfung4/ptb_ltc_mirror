from typing import Union, Dict
from hyperopt import hp
from hyperopt.pyll import scope

from ptb_ltc.config.core import config


def _smote_hp(
    k_neighbors: Union[int, None] = None,
) -> Dict[str, Union[float, int]]:
    """_smote_hp

    Args:
        sampling_strategy (Union[float, None], optional): The sampling strategy to use. Defaults to None.
        random_state (int, optional): The seed used by the random number generator. Defaults to 42.
        k_neighbors (Union[int, None], optional): Number of nearest neighbours to used to construct synthetic samples. Defaults to None.
        n_jobs (Union[int, None], optional): The number of threads to run the algorithm. Defaults to None.

    Returns:
        param_space (Dict): hyperparameter search space for SMOTE
    """
    param_space = dict(
        oversample__k_neighbors=(
            scope.int(hp.quniform("smote_k_neighbors", 1, 10, 1))
            if k_neighbors is None
            else k_neighbors
        ),
    )
    return param_space


def _borderline_smote_hp(
    k_neighbors: Union[int, None] = None,
    m_neighbors: Union[int, None] = None,
) -> Dict[str, Union[float, int]]:
    """_borderline_smote_hp

    Args:
        sampling_strategy (Union[float, None], optional): The sampling strategy to use. Defaults to None.
        random_state (int, optional): The seed used by the random number generator. Defaults to 42.
        k_neighbors (Union[int, None], optional): Number of nearest neighbours to used to construct synthetic samples. Defaults to None.
        m_neighbors (Union[int, None], optional): Number of nearest neighbours to use to determine if a minority sample is in danger. Defaults to None.

    Returns:
        param_space (Dict): hyperparameter search space for BorderlineSMOTE
    """
    param_space = dict(
        oversample__k_neighbors=(
            scope.int(hp.quniform("borderline_smote_k_neighbors", 1, 10, 1))
            if k_neighbors is None
            else k_neighbors
        ),
        oversample__m_neighbors=(
            scope.int(hp.quniform("borderline_smote_m_neighbors", 1, 10, 1))
            if m_neighbors is None
            else m_neighbors
        ),
    )
    return param_space


def _adasyn_hp(
    n_neighbors: Union[int, None] = None,
) -> Dict[str, Union[float, int]]:
    """_adasyn_hp

    Args:
        sampling_strategy (Union[float, None], optional): The sampling strategy to use. Defaults to None.
        n_neighbors (Union[int, None], optional): Number of nearest neighbours to used to construct synthetic samples. Defaults to None.
        n_jobs (Union[int, None], optional): The number of threads to run the algorithm. Defaults to None.

    Returns:
        param_space (Dict): hyperparameter search space for ADASYN
    """
    param_space = dict(
        oversample__n_neighbors=(
            scope.int(hp.quniform("adasyn_n_neighbors", 1, 10, 1))
            if n_neighbors is None
            else n_neighbors
        ),
    )
    return param_space
