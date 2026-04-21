# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: probabl_workshop
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# ## 1 · Load the Stallion dataset
#
# The Stallion dataset records monthly beverage sales across **agencies** (wholesalers /
# stores) and **SKUs** (products). Each row carries the volume sold plus covariates:
# price, discounts, industry-wide volume, weather, demographics, and special-day flags.

# %%
import pandas as pd
DATA = "data/train/"

historical = pd.read_csv(DATA + "historical_volume.csv")
price_promo = pd.read_csv(DATA + "price_sales_promotion.csv")
events = pd.read_csv(DATA + "event_calendar.csv")
industry = pd.read_csv(DATA + "industry_volume.csv")
soda = pd.read_csv(DATA + "industry_soda_sales.csv")
weather = pd.read_csv(DATA + "weather.csv")
demographics = pd.read_csv(DATA + "demographics.csv")
# provided data is for 2017 which would not be available in reality, we just
# use it for illustration assuming that it is quite stable over a few years' range.

# Merge on (Agency, SKU, YearMonth)
df = historical.merge(price_promo, on=["Agency", "SKU", "YearMonth"], how="left")

# Merge on (YearMonth) only
df = df.merge(events, on="YearMonth", how="left")
df = df.merge(industry, on="YearMonth", how="left")
df = df.merge(soda, on="YearMonth", how="left")

# Merge on (YearMonth, Agency)
df = df.merge(weather, on=["YearMonth", "Agency"], how="left")

# Merge on (Agency) only
df = df.merge(demographics, on="Agency", how="left")

# %% [markdown]
# ---
# ## 2 · Data exploration
#
# Use skrub to explore data.
# Some possible questions to answer:
# - How many missing values are there, and in which columns?
# - What are the distributions of numeric features? Are there outliers?
# - What does the target look like?
# - What is the nature of the features? (ordinal, categorical, datetime, text..?)

# %%
from skrub import TableReport
TableReport(df)

# %%
df["YearMonth"] = pd.to_datetime(df["YearMonth"].astype(str), format="%Y%m")

# %%
import plotly.express as px

for agency, df_ in df.groupby("Agency"):
    fig = px.line(df_, x="YearMonth", y="Volume", color="SKU", title=agency)
    fig.show()

# %% [markdown]
# ---
# ## 3 · Feature engineering, build & train pipelines

# %%
def add_demographics(query, data_dir="data/train/"):
    demographics = pd.read_csv(data_dir + "demographics.csv")
    return query.merge(demographics, on="Agency", how="left")

# %%
import skrub
from helpers import add_lagged_volume
from sklearn.dummy import DummyRegressor

def make_data_op(regressor=DummyRegressor()):
    """Build a skrub DataOp pipeline for walk-forward volume forecasting.

    The pipeline:
      1. Sorts the data by Agency / SKU / YearMonth.
      2. Marks YearMonth / Agency / SKU as X and Volume as y.
      3. Adds lagged-volume features (uses the full historical dataset so
         lags are always computed from complete past data, even in CV).
      4. Adds demographic covariates loaded from ``data_dir``.
      5. Vectorises with TableVectorizer.
      6. Applys the regressor.

    Environment keys expected at evaluation time:
        ``data``     – DataFrame with YearMonth, Agency, SKU, Volume.
        ``data_dir`` – Path (str or Path) to the folder containing CSV files.
    """
    data = skrub.var("data")
    X = data[["YearMonth", "Agency", "SKU"]].skb.mark_as_X()
    y = data["Volume"].skb.mark_as_y()
    data_dir = skrub.var("data_dir")

    features = (
        X.skb.apply_func(add_lagged_volume, data)
        .skb.apply_func(add_demographics, data_dir)
        .skb.apply(
            skrub.TableVectorizer(
                datetime=skrub.DatetimeEncoder(resolution="month"),
                low_cardinality=skrub.ToCategorical(),
                cardinality_threshold=100,
            )
        )
    )

    return features.skb.apply(regressor, y=y)


# %%
from sklearn.ensemble import HistGradientBoostingRegressor
# DataOp pipeline: sort → lag features → demographics → TableVectorizer → regressor.
# ``data`` holds the full history so lags are always computed from complete past data,
# even inside a CV fold where X is restricted to the training rows.
data_op = make_data_op(HistGradientBoostingRegressor(categorical_features="from_dtype"))

env = {
    "data": pd.read_csv(DATA + "historical_volume.csv"),
    "data_dir": DATA,
}

data_op

# %% [markdown]
# ---
# ## 4 · Evaluate with Skore
# with Cross Validation

# %%
from helpers import Splitter

splitter = Splitter()

# %% [markdown]
# side by side benchmark, comparison of options (one or two models, with a baseline)

# %% 
import skore

rep_5 = skore.evaluate(data_op, data = env, splitter = 0.1)

# %%
rep_own = skore.evaluate(data_op, data = env, splitter = splitter)

# %%
cv_results = {
    name: make_data_op(name).skb.cross_validate(env, cv=splitter, scoring="neg_mean_absolute_error")
    for name in ["hgb", "prev_month", "dummy"]
}

comparison = (
    pd.DataFrame({name: -res["test_score"] for name, res in cv_results.items()})
    .agg(["mean", "std"])
    .rename(index={"mean": "MAE mean", "std": "MAE std"})
)
comparison

# %% [markdown]
# Evaluation of chosen model
# EstimatorReport of a model trained on the whole training set, and tested on Dec 2017

# %%
from skore import EstimatorReport

last_month = env["data"]["YearMonth"].max()
env_train = {"data": env["data"][env["data"]["YearMonth"] < last_month].copy(), "data_dir": DATA}
env_test  = {"data": env["data"][env["data"]["YearMonth"] == last_month].copy(), "data_dir": DATA}

learner = pred.skb.make_learner().fit(env_train)
report = EstimatorReport(
    learner,
    fit=False,
    X_test=env_test,
    y_test=env_test["data"]["Volume"].reset_index(drop=True),
)
report

# %%
report.metrics.summarize()

# %%
report.metrics.prediction_error()
# %%
