# probabl-workshop
Hands-on experience of skrub, scikit-learn and skore

## Use cases

Thanks to jupytext, transform the python files into notebooks, with the following command:

```bash
jupytext --to notebook embeddings_10k_exploration.py # or stallion_volume_forecasting.py
```

### 1. Stallion Volume Forecasting

End-to-end ML pipeline on the Stallion beverage-sales dataset using **skrub**, **Skore**, and **scikit-learn**.

- Notebook: [`usecase1-stallion_volume_forecasting/stallion_volume_forecasting.ipynb`](usecase1-stallion_volume_forecasting/stallion_volume_forecasting.ipynb)
- Video walkthrough: [`usecase1-stallion_volume_forecasting/recordings/stallion_walkthrough.mp4`](usecase1-stallion_volume_forecasting/recordings/stallion_walkthrough.mp4)

https://github.com/user-attachments/assets/bfbeddea-a068-4f8e-8a49-5cf973dca6d2

To re-record the walkthrough (requires `playwright`):

```bash
cd usecase1-stallion_volume_forecasting/recordings
python record_walkthrough.py            # visible browser
python record_walkthrough.py --headless # headless
```

### 2. 10-K Embeddings — Forward Return Prediction

768-dimensional embeddings of SEC 10-K annual reports (2025, ~4,300 companies) used to **predict 30-day forward stock returns** with **skrub**, **Skore**, and **scikit-learn**.

- Notebook: [`usecase2-embeddings_10k/embeddings_10k_exploration.ipynb`](usecase2-embeddings_10k/embeddings_10k_exploration.ipynb)

**What the notebook covers:**

| Step | Tool | What it does |
|------|------|-------------|
| Load | pandas | Consolidate 234 daily parquet files (year/month/day structure) |
| Explore | `skrub.TableReport` | Filing distribution, GICS sector breakdown |
| Visualize | scikit-learn | PCA → t-SNE projection colored by sector |
| Market data | yfinance | 30-day forward returns after each 10-K filing |
| Modelling | scikit-learn | Ridge, HGBR, RandomForest on PCA-reduced embeddings |
| Evaluation | `skore` | `EstimatorReport`, `CrossValidationReport`, `ComparisonReport` |
| Similarity | cosine distance | Nearest-neighbor company lookup by 10-K content |

**Data**: `data-nlp/embeddings_10k_with_gics/2025/{MM}/{DD}/embeddings_with_gics.parquet` — each row is a company with ticker, GICS sector, and 768-dim embedding of its 10-K filing.

To fetch/refresh market prices (takes ~10 min):

```bash
cd usecase2-embeddings_10k
python fetch_prices.py
```
