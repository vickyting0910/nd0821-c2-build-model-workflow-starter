#!/usr/bin/env python
import argparse
import itertools
import logging
import os

import yaml
import tempfile
import mlflow
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer
import matplotlib.pyplot as plt
import wandb
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn import metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="train_random_forest")
    logger.info("Downloading and reading train artifact")
    train_data_path = run.use_artifact(args.train_data).file()
    df = pd.read_csv(train_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X = df.copy()
    y = X.pop("price")

    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        stratify=df[args.stratify] if args.stratify != "null" else None,
        random_state=args.random_seed,
    )

    logger.info("Setting up pipeline")

    sk_pipe, used_columns = get_training_inference_pipeline(args)

    logger.info("Fitting")
    sk_pipe.fit(X_train[used_columns], y_train)

    # Evaluate
    pred = sk_pipe.predict(X_train[used_columns])

    logger.info("Scoring")
    # AUC
    #score = roc_auc_score(y_val, pred_proba, average="macro", multi_class="ovo")

    # MAE
    n_scores = metrics.mean_absolute_error(y_train,pred)
    run.summary["MAE"] = n_scores

    # Export if required
    logger.info("Export")
    if args.export_artifact != "null":

        export_model(run, sk_pipe, used_columns, X_val, pred, args.export_artifact)
    
    # Some useful plots
    logger.info("plot")
    fig_feat_imp = plot_feature_importance(sk_pipe)
    
    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))

    sub_cm.boxplot(n_scores)
    sub_cm.plot(1, np.mean(n_scores), '-ro',color='red', label='mean_absolute_error')
    #sub_cm.set_ylim([37.5,38.5])
    fig_cm.tight_layout()
    logger.info("plot log")
    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
            "mean_absolute_error": wandb.Image(fig_cm),
        }
    )

def export_model(run, sk_pipe, used_columns, X_val, val_pred, export_artifact):

    # Infer the signature of the model

    # Get the columns that we are really using from the pipeline
    signature = infer_signature(X_val[used_columns], val_pred)

    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "model_export")

        mlflow.sklearn.save_model(
            sk_pipe,
            export_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_val.iloc[:2],
        )

        artifact = wandb.Artifact(
            name=export_artifact,
            type="model_export",
            description="Random Forest pipeline export",
        )
        artifact.add_dir(export_path)
        logger.info("save model")
        run.log_artifact(artifact)

        # Make sure the artifact is uploaded before the temp dir
        # gets deleted
        artifact.wait()


def plot_feature_importance(sk_pipe):

    # We collect the feature importance for all non-nlp features first
    feat_names = np.array(
        sk_pipe["preprocessor"].transformers[0][-1]
        + sk_pipe["preprocessor"].transformers[1][-1]
    )
    feat_imp = sk_pipe["classifier"].feature_importances_[: len(feat_names)]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(sk_pipe["classifier"].feature_importances_[len(feat_names) :])
    feat_imp = np.append(feat_imp, nlp_importance)
    feat_names = np.append(feat_names, "title + song_name")
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp[idx], color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(feat_names[idx], rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_training_inference_pipeline(args):

    # Get the configuration for the pipeline
    with open(args.model_config) as fp:
        model_config = yaml.safe_load(fp)
    # Add it to the W&B configuration so the values for the hyperparams
    # are tracked
    wandb.config.update(model_config)

    # We need 3 separate preprocessing "tracks":
    # - one for categorical features
    # - one for numerical features
    # Categorical preprocessing pipeline
    categorical_features = sorted(model_config["features"]["categorical"])
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder()
    )
    # Numerical preprocessing pipeline
    numeric_features = sorted(model_config["features"]["numerical"])
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )

    # Put the 3 tracks together into one pipeline using the ColumnTransformer
    # This also drops the columns that we are not explicitly transforming
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", non_ordinal_categorical_preproc, categorical_features) 
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    # Get a list of the columns we used
    used_columns = list(itertools.chain.from_iterable([x[2] for x in preprocessor.transformers]))

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestRegressor(**model_config["random_forest"])),
        ]
    )
    return sk_pipe, used_columns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a YAML file containing the configuration for the random forest",
        required=True,
    )

    parser.add_argument(
        "--export_artifact",
        type=str,
        help="Name of the artifact for the exported model. Use 'null' for no export.",
        required=False,
        default="null",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for the random number generator.",
        required=False,
        default=42
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size for the validation set as a fraction of the training set",
        required=False,
        default=0.3
    )

    parser.add_argument(
        "--stratify",
        type=str,
        help="Name of a column to be used for stratified sampling. Default: 'null', i.e., no stratification",
        required=False,
        default="null",
    )

    args = parser.parse_args()

    go(args)
