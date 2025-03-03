import scipy.stats
import pandas as pd
import numpy as np

def test_column_presence_and_type(data):

    # Disregard the reference dataset
    _, data = data

    required_columns = {
        "latitude": pd.api.types.is_float_dtype,
        "longitude": pd.api.types.is_float_dtype,
        "price": pd.api.types.is_integer_dtype
    }

    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(data[col_name]), f"Column {col_name} failed test {format_verification_funct}"


def test_class_names(data):

    # Disregard the reference dataset
    _, data = data

    # Check that only the known classes are present
    known_classes = [
        "Manhattan",
        "Brooklyn",
        "Queens",
        "Bronx",
        "Staten Island"
    ]

    assert data["neighbourhood_group"].isin(known_classes).all()


def test_column_ranges(data):

    # Disregard the reference dataset
    _, data = data

    ranges = {
        "latitude": (40.5, 41.2),
        "longitude": (-74.25, -73.50)
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert data[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={data[col_name].min()} and max={data[col_name].max()}"
        )


def test_kolmogorov_smirnov(data, ks_alpha):

    sample1, sample2 = data

    columns = [
        "price"
    ]

    # Bonferroni correction for multiple hypothesis testing
    # (see my blog post on this topic to see where this comes from:
    # https://towardsdatascience.com/precision-and-recall-trade-off-and-multiple-hypothesis-testing-family-wise-error-rate-vs-false-71a85057ca2b)
    alpha_prime = 1 - (1 - ks_alpha)**(1 / len(columns))

    for col in columns:

        ts, p_value = scipy.stats.ks_2samp(sample1[col], sample2[col])

        # NOTE: as always, the p-value should be interpreted as the probability of
        # obtaining a test statistic (TS) equal or more extreme that the one we got
        # by chance, when the null hypothesis is true. If this probability is not
        # large enough, this dataset should be looked at carefully, hence we fail
        assert p_value < alpha_prime

def test_row_count(data: pd.DataFrame):
    """
    Test row count before and after processing
    """
    sample1, sample2 = data
    assert sample1.shape[0] != sample2.shape[0]

def test_price_range(data: pd.DataFrame, min_price, max_price):
    """
    Test the range of price
    """
    data, _ = data
    idx = data['price'].between(min_price, max_price)
    assert np.sum(~idx) == 0

