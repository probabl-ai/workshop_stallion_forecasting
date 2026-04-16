# probabl-workshop
Hands-on experience of skrub, scikit-learn and skore

## Use cases

Thanks to jupytext, transform the python files into notebooks, with the following command:

```bash
jupytext --to notebook stallion_volume_forecasting.py
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