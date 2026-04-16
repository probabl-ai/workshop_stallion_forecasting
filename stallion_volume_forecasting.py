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
import numpy as np
import pandas as pd

# %%
DATA = "data/train/"

historical = pd.read_csv(DATA + "historical_volume.csv")
price_promo = pd.read_csv(DATA + "price_sales_promotion.csv")
events = pd.read_csv(DATA + "event_calendar.csv")
industry = pd.read_csv(DATA + "industry_volume.csv")
soda = pd.read_csv(DATA + "industry_soda_sales.csv")
weather = pd.read_csv(DATA + "weather.csv")
demographics = pd.read_csv(DATA + "demographics.csv")

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

# %% [markdown]
# ---
# ## 3 · Feature engineering, build & train pipelines

# %%
from skrub import TableVectorizer, tabular_pipeline
pipeline = tabular_pipeline("regression")
pipeline

# %% [markdown]
# ---
# ## 4 · Evaluate with Skore
# with Cross Validation

# %%
from skore import CrossValidationReport, evaluate

# %% [markdown]
# side by side benchmark, comparison of options (one or two models, with a baseline)

# %%
from skore import ComparisonReport


# %% [markdown]
# Evaluation of chosen model
# EstimatorReport of a model trained on the whole training set, and tested on Jan 18

# %%
from skore import ComparisonReport