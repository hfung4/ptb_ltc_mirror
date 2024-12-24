"""
Model validation
* Compare model performance (measured by a set of evaluation metric) to the "champion" model in the dev environment.  
* We should also have read access to the production workspace, which allow us to compare the performance of the challenger model to the production model.
* Transition the "challenger" model to "champion" in the development workspace if the challenger model has a better performance 
"""

import mlflow
from mlflow import MlflowClient
from argparse import ArgumentParser

from ptb_ltc.config.core import config
import ptb_ltc.deploy.deploy_utils_ws as du
from ptb_ltc.logging import logger

# Get user inputs and set parameters
parser = ArgumentParser(description="Run mode and deploy environment.")
parser.add_argument(
    "--run_mode",
    type=str,
    default="dry_run",
    choices=["disabled", "dry_run", "enabled"],
    help="The mode in which the script validate.py runs (default: 'dry_run').",
)
parser.add_argument(
    "--env",
    type=str,
    required=True,
    choices=["dev", "test", "staging", "prod"],
    help="The deploy environment name (e.g., 'dev', 'test', 'staging', 'prod').",
)

try:
    args = parser.parse_args()
    run_mode = args.run_mode
    env = args.env
except SystemExit as e:
    logger.error("Error: Invalid arguments. Use --help for usage information.")
    raise e

run_mode = args.run_mode
env = args.env

full_model_name = f"{env}_{config.model.MODEL_NAME}"

if run_mode != "disabled":
    # Get the registered model
    model = du.get_registered_model(full_model_name)

    client = MlflowClient()

    # Get challenger run
    challenger_mv = du.get_model_version_by_stage_and_alias(model, "None")
    challenger_run_id = challenger_mv.run_id
    challenger_run = du.get_run_from_run_id(challenger_run_id)

    # Get comparison metrics for the model from the challenger_run
    challenger_metrics = du.get_comparison_metrics(
        run=challenger_run,
        metric_names=["test_log_loss", "test_roc_auc", "test_auprc_lift"],
        larger_is_better=[False, True, True],
    )
    logger.info(f"Challenger model metrics: {challenger_metrics}")

    try:
        # Get champion run
        champion_mv = du.get_model_version_by_stage_and_alias(
            model, "Staging", "Champion"
        )
        champion_run_id = champion_mv.run_id
        champion_run = du.get_run_from_run_id(champion_run_id)

        # Get comparison metrics for the model from the challenger_run
        champion_metrics = du.get_comparison_metrics(
            run=champion_run,
            metric_names=["test_log_loss", "test_roc_auc", "test_auprc_lift"],
            larger_is_better=[False, True, True],
        )
        logger.info(f"Champion model metrics: {champion_metrics}")

        # Compare the performance of challenger and champion models
        comparison_result = du.compare_metrics(challenger_metrics, champion_metrics)
        logger.info(f"Challenger outperforms Champion: {comparison_result}")

        # If challenger outperforms champion in terms of all metrics, or run_mode is 'dry_run'
        if (all(comparison_result.values())) or (run_mode == "dry_run"):
            # Transition challenger to champion
            du.transition_model(challenger_mv, "Staging")
            # Remove tag of previous champion
            client.delete_model_version_tag(
                champion_mv.name, champion_mv.version, "alias"
            )
            logger.info(f"Challenger is promoted to Champion.")
    except ValueError as e:
        # Handles the case when no model version is found in the 'Staging' stage with the alias 'Champion'
        if "No model version found in stage 'Staging' with alias 'Champion'" in str(e):
            logger.error(
                "No Champion model found in 'Staging'. Promoting the challenger if it meets the thresholds."
            )

            if (
                (challenger_metrics["test_roc_auc"] >= config.model.MIN_TEST_ROC_AUC)
                or (
                    challenger_metrics["test_auprc_lift"]
                    >= config.model.MIN_TEST_AURPC_LIFT
                )
                or run_mode == "dry_run"
            ):
                # Accept the current model and promote it to "Champion" if its metrics meet the minimum threshold or run_mode is 'dry_run'
                du.transition_model(challenger_mv, "Staging")
                logger.info(f"Current model is promoted to the first Champion.")
            else:
                logger.info(
                    f"Rejected the current model since its metrics do not meet the minimum threshold of TEST_ROC_AUC >= {config.model.MIN_TEST_ROC_AUC} or TEST_AUROC >= {config.model.MIN_TEST_AURPC_LIFT}"
                )
        else:
            # Re-raise the exception if it's not the specific error we're catching
            raise
