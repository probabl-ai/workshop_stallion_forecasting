import itertools
from pathlib import Path
import datetime

import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
import sklearn.pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import RegressorMixin, BaseEstimator
import polars as pl


def project_dir():
    return Path(__file__).parent


def data_dir():
    return project_dir() / "data"


def test_data_dir():
    return data_dir() / "test_8uviCCm"


def train_data_dir():
    return data_dir() / "train_OwBvO8W"


def results_dir(subdir=None, /):
    results_dir = project_dir() / "results"
    if subdir:
        results_dir = results_dir / subdir
    results_dir.mkdir(exist_ok=True, parents=True)
    return results_dir


def year_month_to_date(df):
    if "YearMonth" not in df.columns:
        return df
    return df.with_columns(pl.col("YearMonth").cast(str).str.to_date("%Y%m"))


def load(csv):
    return year_month_to_date(pl.read_csv(str(csv)))


def sort_data(df):
    return df.sort("Agency", "SKU", "YearMonth")


def add_lagged_volume(query, historical_data_dir):
    history = load(historical_data_dir / "historical_volume.csv")
    unseen = query.join(history, on=("YearMonth", "Agency", "SKU"), how="anti").select(
        "Agency", "SKU", "YearMonth", pl.lit(None).alias("Volume")
    )
    history = pl.concat([history, unseen], how="vertical").sort("YearMonth")
    for lag, window in [(1, 1), (2, 1), (3, 1), (12, 1), (1, 3), (12, 3)]:
        assert lag >= 1
        col_name = f"Volume_lag_{lag}mo_window_{window}mo"
        hist = history.rolling(
            index_column="YearMonth",
            group_by=("Agency", "SKU"),
            offset=f"-{lag + window}mo",
            period=f"{window}mo",
        ).agg(pl.first("Volume").alias(col_name))
        query = (
            query.join(
                hist,
                on=("YearMonth", "Agency", "SKU"),
                how="left",
                maintain_order="left",
            )
            .join(
                hist.group_by("YearMonth", "Agency").agg(
                    pl.sum(col_name).alias(f"Agency_{col_name}")
                ),
                on=("YearMonth", "Agency"),
                how="left",
                maintain_order="left",
            )
            .join(
                hist.group_by("YearMonth", "SKU").agg(
                    pl.sum(col_name).alias(f"SKU_{col_name}")
                ),
                on=("YearMonth", "SKU"),
                how="left",
                maintain_order="left",
            )
        )
    return query


def add_demographics(query, historical_data_dir):
    """Add demographics info.

    provided data is for 2017 which would not be available in reality, we just
    use it for illustration assuming that it is quite stable over a few years' range.
    """
    demographics = load(historical_data_dir / "demographics.csv")
    return query.join(demographics, on="Agency", how="left", maintain_order="left")


class PrevMonth(RegressorMixin, BaseEstimator):
    """Basic baseline that repeats the previous month."""

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X["Volume_lag_1mo_window_1mo"]


class Splitter:
    def __init__(self, interval=1):
        self.interval = interval

    def split(self, X, y=None, groups=None):
        min_train_months = 6
        start = X.select(
            pl.col("YearMonth").min().dt.offset_by(f"{min_train_months}mo")
        )["YearMonth"][0]
        stop = X["YearMonth"].max()
        test_months = pl.date_range(
            start, stop, interval=f"{self.interval}mo", closed="both", eager=True
        )
        for test_mo in test_months:
            train = (
                X.with_row_index()
                .filter(pl.col("YearMonth") < test_mo)["index"]
                .to_numpy()
            )
            test = (
                X.with_row_index()
                .filter(pl.col("YearMonth") == test_mo)["index"]
                .to_numpy()
            )
            if train.shape[0] and test.shape[0]:
                yield train, test

    def get_n_splits(self, X, y=None, groups=None):
        return sum(1 for _ in self.split(X, y=y, groups=groups))


def make_data_op(regressor="hgb"):
    data = skrub.var("data").skb.apply_func(sort_data)
    X = data[["YearMonth", "Agency", "SKU"]].skb.mark_as_X(cv=Splitter())
    y = data["Volume"].skb.mark_as_y()
    historical_data_dir = skrub.var("historical_data_dir")
    features = (
        X.skb.apply_func(add_lagged_volume, historical_data_dir)
        .skb.apply_func(add_demographics, historical_data_dir)
        .skb.apply(
            skrub.TableVectorizer(
                datetime=skrub.DatetimeEncoder(resolution="month"),
                low_cardinality=skrub.ToCategorical(),
                cardinality_threshold=100,
            )
        )
    )
    hgb = HistGradientBoostingRegressor(categorical_features="from_dtype")
    regressor = {"hgb": hgb, "dummy": DummyRegressor(), "prev_month": PrevMonth()}[
        regressor
    ]
    pred = features.skb.apply(regressor, y=y).skb.with_scoring(
        "neg_mean_absolute_error"
    )
    return pred


def make_pipeline(regressor="hgb", historical_data_dir=None):
    if historical_data_dir is None:
        historical_data_dir = train_data_dir()
    hgb = HistGradientBoostingRegressor(categorical_features="from_dtype")
    regressor = {"hgb": hgb, "dummy": DummyRegressor(), "prev_month": PrevMonth()}[
        regressor
    ]
    return sklearn.pipeline.make_pipeline(
        FunctionTransformer(
            add_lagged_volume, kw_args={"historical_data_dir": historical_data_dir}
        ),
        FunctionTransformer(
            add_demographics, kw_args={"historical_data_dir": historical_data_dir}
        ),
        skrub.TableVectorizer(
            datetime=skrub.DatetimeEncoder(resolution="month"),
            low_cardinality=skrub.ToCategorical(),
            cardinality_threshold=100,
        ),
        regressor,
    )


def get_env():
    history_file = train_data_dir() / "historical_volume.csv"
    return {
        "data": load(history_file),
        "historical_data_dir": train_data_dir(),
    }


def get_test_env():
    query_file = test_data_dir() / "volume_forecast.csv"
    data = load(query_file).select(
        "Agency", "SKU", pl.lit("2018-01-01").str.to_date().alias("YearMonth")
    )
    return {"data": data, "historical_data_dir": train_data_dir()}


def get_Xy():
    data = get_env()["data"]
    return data[["YearMonth", "Agency", "SKU"]], data["Volume"]


def get_test_X():
    data = get_test_env()["data"]
    return data[["YearMonth", "Agency", "SKU"]]
