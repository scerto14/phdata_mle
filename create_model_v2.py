# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 11:03:42 2025

@author: Salce
"""

import json
import pathlib
import pickle
from typing import List
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
import xgboost as xgb


SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/kc_house_data.csv"  # path to CSV with demographics
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str, demographics_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pd.read_csv(sales_path, dtype={'zipcode': str})
    demographics = pd.read_csv("data/zipcode_demographics.csv",
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left", on="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    
    merged_data.drop(['id','lat','long','urbn_ppltn_qty',
    'sbrbn_ppltn_qty', 'farm_ppltn_qty', 'non_farm_qty', 'edctn_less_than_9_qty', 'edctn_9_12_qty', 'edctn_high_schl_qty',
    'edctn_some_clg_qty', 'edctn_assoc_dgre_qty', 'edctn_bchlr_dgre_qty',
    'edctn_prfsnl_qty'], axis=1, inplace=True)

    #Feature Engineering
    merged_data['year'] = pd.to_datetime(merged_data['date']).apply(lambda x: x.year)
    merged_data['month'] = pd.to_datetime(merged_data['date']).apply(lambda x: x.month)
    merged_data['day'] = pd.to_datetime(merged_data['date']).apply(lambda x: x.day)
    merged_data['renovated'] = merged_data['yr_renovated'].apply(lambda x: 1 if x>0 else 0)
    merged_data['age_of_house'] = merged_data['year'] - merged_data['yr_built']
    merged_data['yrs_since_reno'] = [min(row['year']-row['yr_renovated'],  row['year']-row['yr_built']) for _, row in merged_data.iterrows()]

    merged_data.drop(['date','zipcode'], inplace=True, axis=1)
    x = merged_data

    return x, y


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
        x, y, random_state=42)

    model = xgb.XGBRegressor(max_depth=5, colsample_bytree=.8, gamma=0, learning_rate=.15, min_child_weight=3)
    model.fit(x_train, y_train)


    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(model, open(output_dir / "model_v2.pkl", 'wb'))
    json.dump(list(x_train.columns),
              open(output_dir / "model_v2_features.json", 'w'))


if __name__ == "__main__":
    main()
