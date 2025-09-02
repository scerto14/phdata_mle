# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 07:36:16 2025

@author: Salce
"""

import requests
import pandas as pd


#Local testing - docker
data = pd.read_csv('data/future_unseen_examples.csv',dtype={'zipcode': str})
payload = data.iloc[0,:].to_dict()
resp = requests.post('http://127.0.0.1:8000/predict_single', json=payload)
print(f"Status Code: {resp.status_code}")
print(f"Response: {resp.json()}")


payload = data.to_dict(orient='records')
resp = requests.post('http://127.0.0.1:8000/predict_multi', json=payload)
print(f"Status Code: {resp.status_code}")
print(f"Response: {resp.json()}")

resp = requests.get('http://127.0.0.1:8000/health')
print(f"Status Code: {resp.status_code}")
print(f"Response: {resp.json()}")

#Load new model
resp = requests.post('http://127.0.0.1:8000/load_new_model', params = {'new_model_name': 'model_retrained.pkl', 'new_model_version':'v2'})
print(f"Status Code: {resp.status_code}")
print(f"Response: {resp.json()}")

#Test new model
payload = data.iloc[0,:].to_dict()
resp = requests.post('http://127.0.0.1:8000/predict_single', json=payload)
print(f"Status Code: {resp.status_code}")
print(f"Response: {resp.json()}")

#Load original model
resp = requests.post('http://127.0.0.1:8000/load_new_model', params = {'new_model_name': 'model.pkl', 'new_model_version':'v1'})
print(f"Status Code: {resp.status_code}")
print(f"Response: {resp.json()}")
