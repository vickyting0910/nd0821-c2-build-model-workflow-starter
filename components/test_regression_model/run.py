#!/usr/bin/env python
import argparse
import itertools
import logging
import pandas as pd
import wandb
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_regression_model")

    logger.info("Downloading and reading test artifact")
    test_data_path = run.use_artifact(args.test_data).file()
    df = pd.read_csv(test_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X_test = df.copy()
    y_test = X_test.pop("neighbourhood_group")

    logger.info("Downloading and reading the exported model")
    model_export_path = run.use_artifact(args.model_export).download()

    pipe = mlflow.sklearn.load_model(model_export_path)

    used_columns = list(itertools.chain.from_iterable([x[2] for x in pipe['preprocessor'].transformers]))
    pred_proba = pipe.predict_proba(X_test[used_columns])

    logger.info("Scoring")
    # MAE
    n_scores = metrics.mean_absolute_error(y_val,pred)
    run.summary["MAE"] = n_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True,
    )

    args = parser.parse_args()

    go(args)
