import pytest
import pandas as pd
import wandb
import numpy as np

run = wandb.init(job_type="data_check")


def pytest_addoption(parser):
    parser.addoption("--reference_artifact", action="store")
    parser.addoption("--sample_artifact", action="store")
    parser.addoption("--ks_alpha", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope="session")
def data(request):

    reference_artifact = request.config.option.reference_artifact

    if reference_artifact is None:
        pytest.fail("--reference_artifact missing on command line")

    sample_artifact = request.config.option.sample_artifact

    if sample_artifact is None:
        pytest.fail("--sample_artifact missing on command line")

    local_path = run.use_artifact(reference_artifact).file()
    sample1 = pd.read_csv(local_path)

    local_path = run.use_artifact(sample_artifact).file()
    sample2 = pd.read_csv(local_path)

    return sample1, sample2


@pytest.fixture(scope='session')
def ks_alpha(request):
    ks_alpha = request.config.option.ks_alpha

    if ks_alpha is None:
        pytest.fail("--ks_threshold missing on command line")

    return float(ks_alpha)

@pytest.fixture(scope='session')
def min_price(request):
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("--min_price_threshold missing on command line")

    return float(min_price)

@pytest.fixture(scope='session')
def max_price(request):
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("--max_price_threshold missing on command line")

    return float(max_price)
