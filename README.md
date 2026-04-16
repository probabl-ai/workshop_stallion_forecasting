# probabl-workshop
Hands-on experience of skrub, scikit-learn and skore.

## How-to

This repository contains a draft of python file, that can be converted in to a notebook, to treat the use-case of Stallion. The data comes from this fictional situation described in a [kaggle competition](https://www.kaggle.com/datasets/utathya/future-volume-prediction).

The script contains already some elements, and we invite you to complete it.

### Technical precisions

Thanks to jupytext, transform the python files into notebooks, with the following command:

```bash
jupytext --to notebook stallion_volume_forecasting.py
```

All the necessary libraries are listed in the requirements.

## Goal

Based on the data from previous months, we want to predict the sales for the following one (testing is for Jan 18 only), for each agency and SKU.

## Dataset fictional context

Country Beeristan, a high potential market, accounts for nearly 10% of Stallion & Co.’s global beer sales. Stallion & Co. has a large portfolio of products distributed to retailers through wholesalers (agencies). There are thousands of unique wholesaler-SKU/products combinations. In order to plan its production and distribution as well as help wholesalers with their planning, it is important for Stallion & Co. to have an accurate estimate of demand at SKU level for each wholesaler.

Currently demand is estimated by sales executives, who generally have a “feel” for the market and predict the net effect of forces of supply, demand and other external factors based on past experience. The more experienced a sales exec is in a particular market, the better a job he does at estimating. Joshua, the new Head of S&OP for Stallion & Co. just took an analytics course and realized he can do the forecasts in a much more effective way. He approaches you, the best data scientist at Stallion, to transform the exercise of demand forecasting.

