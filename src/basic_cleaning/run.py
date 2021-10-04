#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import logging
import os

import wandb
import pandas as pd

from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="process_data")
    
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    # Drop the duplicates
    logger.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)

    # A minimal feature engineering step: a new feature
    logger.info("Feature engineering")
    df['name'].fillna(value='', inplace=True)
    df['host_name'].fillna(value='', inplace=True)
    df['text_feature'] = df['name'] + ' ' + df['host_name']
    df=df.fillna(0.0).replace('',0.0)
    # Drop outliers
    min_price = float(args.min_price)
    max_price = float(args.max_price)
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    filename = "processed_data.csv"
    df.to_csv(filename)

    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")
    log_artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
        filename,
        run,
    )
    os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URL to a local destination")

    parser.add_argument("input_artifact", type=str, help="path")

    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")

    parser.add_argument("artifact_type", type=str, help="Output artifact type.")

    parser.add_argument(
        "artifact_description", type=str, help="A brief description of this artifact"
    )

    parser.add_argument("min_price", type=str, help="min price threshold")

    parser.add_argument("max_price", type=str, help="max price threshold")

    args = parser.parse_args()

    go(args)
