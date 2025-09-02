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
from sklearn.metrics import mean_squared_error, PredictionErrorDisplay, r2_score
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import matplotlib.pyplot as plt

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/kc_house_data.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
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
    data = pd.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pd.read_csv("data/zipcode_demographics.csv",
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y


#Existing model metrics
x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
x_train, _x_test, y_train, _y_test = model_selection.train_test_split(x, y, random_state=42)

model = pipeline.make_pipeline(preprocessing.RobustScaler(),
                               neighbors.KNeighborsRegressor()).fit(
                                   x_train, y_train)


model_v1_train_rmse = mean_squared_error(y_train, model.predict(x_train), squared=False)
model_v1_test_rmse = mean_squared_error(_y_test, model.predict(_x_test), squared=False)
model_v1_r2 = r2_score(_y_test, model.predict(_x_test))


display = PredictionErrorDisplay.from_predictions(y_true=np.log(_y_test), y_pred=np.log(model.predict(_x_test)), subsample=None)

model_v1_preds = model.predict(_x_test)


plt.title('Model V1 Test Residuals')
plt.show()



#New model exploration
data = pd.read_csv(SALES_PATH, dtype={'zipcode': str})
demographics = pd.read_csv("data/zipcode_demographics.csv",
                               dtype={'zipcode': str})

merged_data = data.merge(demographics, how="left",
                         on="zipcode")

y = merged_data.pop('price')
merged_data.drop(['id','lat','long','urbn_ppltn_qty',
'sbrbn_ppltn_qty', 'farm_ppltn_qty', 'non_farm_qty', 'edctn_less_than_9_qty', 'edctn_9_12_qty', 'edctn_high_schl_qty',
'edctn_some_clg_qty', 'edctn_assoc_dgre_qty', 'edctn_bchlr_dgre_qty',
'edctn_prfsnl_qty'], axis=1, inplace=True)

merged_data['year'] = pd.to_datetime(merged_data['date']).apply(lambda x: x.year)
merged_data['month'] = pd.to_datetime(merged_data['date']).apply(lambda x: x.month)
merged_data['day'] = pd.to_datetime(merged_data['date']).apply(lambda x: x.day)
merged_data['renovated'] = merged_data['yr_renovated'].apply(lambda x: 1 if x>0 else 0)
merged_data['age_of_house'] = merged_data['year'] - merged_data['yr_built']

merged_data['yrs_since_reno'] = [min(row['year']-row['yr_renovated'],  row['year']-row['yr_built']) for _, row in merged_data.iterrows()]

merged_data.drop(['date','zipcode'], inplace=True, axis=1)
x = merged_data.values

x_train, _x_test, y_train, _y_test = model_selection.train_test_split(x, y, random_state=42)


# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'learning_rate': [0.05, 0.1, 0.15],
#     'max_depth': [3, 4, 5, 6, 8],
#     'min_child_weight': [1, 3, 5, 7],
#     'gamma': [0.0, 0.1, 0.2],
#     'colsample_bytree': [0.3, 0.4,.6,.8]
# }


# clf = GridSearchCV(xgb.XGBRegressor(), param_grid=param_grid, scoring='neg_root_mean_squared_error')

# clf.fit(x_train,y_train)

# clf.best_params_

clf = xgb.XGBRegressor(max_depth=5, colsample_bytree=.8, gamma=0, learning_rate=.15, min_child_weight=3)
clf.fit(x_train, y_train)

model_v2_train_rmse = mean_squared_error(y_train, clf.predict(x_train), squared=False)
model_v2_test_rmse = mean_squared_error(_y_test, clf.predict(_x_test), squared=False)
model_v2_r2 = r2_score(_y_test, clf.predict(_x_test))



print('Train RMSE v1 model ', model_v1_train_rmse)
print('Train RMSE v2 model ', model_v2_train_rmse)
print('Test RMSE v1 model ', model_v1_test_rmse)
print('Test RMSE v2 model ', model_v2_test_rmse)
print('Test R2 v1 model ', model_v1_r2)
print('Test R2 v2 model ', model_v2_r2)

      
display = PredictionErrorDisplay.from_predictions(y_true=np.log(_y_test), y_pred=np.log(clf.predict(_x_test)), subsample=None)
plt.title('Model V2 Test Residuals')
plt.show()


model_v2_preds = clf.predict(_x_test)

resultsDF = pd.DataFrame(columns = ['v1_pred_error','v2_pred_error'])
resultsDF['v1_pred_error'] = _y_test - model_v1_preds 
resultsDF['v2_pred_error'] = _y_test - model_v2_preds 

resultsDF = resultsDF.melt()


import seaborn as sns
sns.boxplot(resultsDF, x='variable', y='value', showfliers=False)


plt.scatter(_y_test, _y_test- model_v1_preds, label='v1_model', alpha=.5)
plt.scatter(_y_test, _y_test- model_v2_preds, label='v2_model', alpha=.5)

plt.legend()
plt.title('Model Comparisons')
plt.show()


plt.scatter(_y_test, model_v1_preds, label='v1_model', alpha=.5)
plt.scatter(_y_test, model_v2_preds, label='v2_model', alpha=.5)
plt.legend()
plt.title('Model Comparisons')
plt.show()