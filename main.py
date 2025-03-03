import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "train_val_test_split",
    "train_random_forest",
    "test_regression_model"
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
		os.path.join(root_path, "components/get_data"),
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "sample",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
	    os.path.join(root_path, "components/basic_cleaning"),
            "main",
            parameters={
                "input_artifact": "sample.csv:latest",
                "artifact_name": "clean_data.csv",
                "artifact_type": "clean_data",
                "artifact_description": "Data with preprocessing applied",
		"min_price": config['data_check']['min_price'],
                "max_price": config['data_check']['max_price']
            },
        )

        if "data_check" in active_steps:
            _ = mlflow.run(
            os.path.join(root_path, "components/data_check"),
            "main",
            parameters={
                "reference_artifact": "clean_data.csv:latest",
                "sample_artifact": "sample.csv:latest",
                "ks_alpha": config["data_check"]["kl_threshold"],
		"min_price": config['data_check']['min_price'],
                "max_price": config['data_check']['max_price'],
            },
        )

        if "train_val_test_split" in active_steps:
            _ = mlflow.run(
            os.path.join(root_path, "components/train_val_test_split"),
            "main",
            parameters={
                "input_artifact": "clean_data.csv:latest",
                "artifact_root": "data",
                "artifact_type": "segregated_data",
                "test_size": config["modeling"]["test_size"],
                "stratify": config["modeling"]["stratify_by"]
            },
        )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                fp.write(OmegaConf.to_yaml(config["modeling"]))

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step
            _ = mlflow.run(
            os.path.join(root_path, "components/train_random_forest"),
            "main",
            parameters={
                "train_data": "data_train.csv:latest",
                "model_config": rf_config,
                "export_artifact": config["modeling"]["export_artifact"],
                "random_seed": config["modeling"]["random_seed"],
                "val_size": config["modeling"]["test_size"],
                "stratify": config["modeling"]["stratify_by"]
            },
        )

        if "test_regression_model" in active_steps:

            _ = mlflow.run(
            os.path.join(root_path, "components/test_regression_model"),
            "main",
            parameters={
                "model_export": f"{config['modeling']['export_artifact']}:prod_training",
                "test_data": "data_test.csv:latest"
            },
        )


if __name__ == "__main__":
    go()
