import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator

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
        ym = pd.to_datetime(X["YearMonth"])
        start = ym.min() + pd.DateOffset(months=min_train_months)
        test_months = pd.date_range(start, ym.max(), freq="MS")[:: self.interval]
        for test_mo in test_months:
            train = X.index[ym < test_mo].to_numpy()
            test = X.index[ym == test_mo].to_numpy()
            if len(train) and len(test):
                yield train, test

    def get_n_splits(self, X, y=None, groups=None):
        return sum(1 for _ in self.split(X, y=y, groups=groups))

def add_lagged_volume(query, history):
    """Add lagged volume features to query rows.

    Parameters
    ----------
    query : DataFrame with at least YearMonth, Agency, SKU columns.
    history : DataFrame with YearMonth, Agency, SKU, Volume columns.
        Full historical volume used to compute lags. Passed as the
        unsplit ``data`` variable in the DataOp so it always covers the
        complete training period regardless of the CV fold.
    """
    history = history[["YearMonth", "Agency", "SKU", "Volume"]].copy()

    # Reset index so boolean masks from the merge stay aligned with query
    query = query.reset_index(drop=True)

    # Rows in query not yet in history (e.g. future forecast dates)
    merged = query[["YearMonth", "Agency", "SKU"]].merge(
        history[["YearMonth", "Agency", "SKU"]],
        on=["YearMonth", "Agency", "SKU"],
        how="left",
        indicator=True,
    )
    unseen = query.loc[merged["_merge"] == "left_only", ["YearMonth", "Agency", "SKU"]].copy()
    unseen["Volume"] = float("nan")

    history_ext = (
        pd.concat([history, unseen], ignore_index=True)
        .sort_values(["Agency", "SKU", "YearMonth"])
        .reset_index(drop=True)
    )

    result = query[["YearMonth", "Agency", "SKU"]].copy().reset_index(drop=True)

    for lag, window in [(1, 1), (2, 1), (3, 1), (12, 1), (1, 3), (12, 3)]:
        assert lag >= 1
        col_name = f"Volume_lag_{lag}mo_window_{window}mo"

        lag_df = history_ext.copy()
        lag_df[col_name] = history_ext.groupby(["Agency", "SKU"])["Volume"].transform(
            lambda x, l=lag, w=window: x.shift(l).rolling(w, min_periods=1).mean()
        )

        result = result.merge(
            lag_df[["YearMonth", "Agency", "SKU", col_name]],
            on=["YearMonth", "Agency", "SKU"],
            how="left",
        )

        agency_agg = (
            lag_df.groupby(["YearMonth", "Agency"])[col_name]
            .sum()
            .rename(f"Agency_{col_name}")
            .reset_index()
        )
        result = result.merge(agency_agg, on=["YearMonth", "Agency"], how="left")

        sku_agg = (
            lag_df.groupby(["YearMonth", "SKU"])[col_name]
            .sum()
            .rename(f"SKU_{col_name}")
            .reset_index()
        )
        result = result.merge(sku_agg, on=["YearMonth", "SKU"], how="left")

    return result
