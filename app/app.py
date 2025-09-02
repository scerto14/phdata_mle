# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:11:22 2025

@author: Salce
"""

import pickle
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated, get_type_hints
import threading 
import os
import redis
import datetime

app = FastAPI()

print('server started')
REDIS_HOST = os.environ.get('REDIS_HOST', '127.0.0.1')
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0)

lock = threading.Lock()
MODEL_KEY = "house_price_model"

with open('model/model_features.json', 'r') as f:
    features = json.load(f)


model_data = redis_client.get(MODEL_KEY)
if model_data:
    model = pickle.loads(model_data)
else:
    model = pickle.load(open('model/model.pkl', 'rb'))
    redis_client.set(MODEL_KEY, pickle.dumps(model))
    redis_client.set('model_version','v1')


    
demographics = pd.read_csv('data/zipcode_demographics.csv', dtype={'zipcode': str})

lock = threading.Lock()

class HouseInput(BaseModel):  
    bedrooms: float
    bathrooms: float
    sqft_living: Annotated[float, Field(gt=0)]
    sqft_lot: Annotated[float, Field(gt=0)]
    floors: Annotated[float, Field(ge=1)]
    waterfront: Annotated[float, Field(ge=0, le=1, multiple_of=1)] = 0.0
    view: Annotated[float, Field(ge=0)]
    condition: Annotated[float, Field(ge=1)]
    grade: Annotated[float, Field(ge=1)]
    sqft_above: Annotated[float, Field(gt=0)]
    sqft_basement: Annotated[float, Field(ge=0)]
    yr_built: Annotated[float, Field(gt=0)]
    yr_renovated: Annotated[float, Field(ge=0)]
    zipcode: str = Field(min_length=5, max_length=5)
    lat: float
    long: float
    sqft_living15: Annotated[float, Field(gt=0)]
    sqft_lot15: Annotated[float, Field(gt=0)]
    

class HouseInputMinimal(BaseModel):  
    bedrooms: float
    bathrooms: float
    sqft_living: Annotated[float, Field(gt=0)]
    sqft_lot: Annotated[float, Field(gt=0)]
    floors: Annotated[float, Field(ge=1)]
    sqft_above: Annotated[float, Field(gt=0)]
    sqft_basement: Annotated[float, Field(ge=0)]
    zipcode: str = Field(min_length=5, max_length=5)
    

def update_model(message):
    global model
    with lock:
        print('loading new model')
        new_model = pickle.loads(redis_client.get(MODEL_KEY)) 
        model = new_model

pubsub = redis_client.pubsub()
pubsub.subscribe(**{'model_update': update_model})
pubsub.run_in_thread(sleep_time=0.001)


@app.post("/predict_single")
async def predict(input_data: HouseInput):

    with lock:
        df = pd.DataFrame([input_data.dict()])
        expected_types = get_type_hints(HouseInput)
        
        df = df.astype(expected_types)
        df = df.merge(demographics, on='zipcode', how='left')
        
        input_features = df[features]
        pred = model.predict(input_features)[0]
    
    return {"prediction": pred, "metadata": {"model_version": redis_client.get('model_version'), 'input_data_type':'full', 'input_data_size':'single'}}

@app.post("/predict_min_data")
async def predict_minimal(input_data: HouseInputMinimal):

    with lock:
        df = pd.DataFrame([input_data.dict()])
        expected_types = get_type_hints(HouseInput)
        
        df = df.astype(expected_types)
        df = df.merge(demographics, on='zipcode', how='left')
        
        input_features = df[features]
        pred = model.predict(input_features)[0]
    
    return {"prediction": pred, "metadata": {"model_version": redis_client.get('model_version'), 'input_data_type':'minimal', 'input_data_size':'single'}}


@app.post("/predict_multi")
async def predict_multi(input_data: list[HouseInput]):
#Or, alternatively:
#@app.post("/predict_multi/<tableName>")
# async def predict_multi(tableName):
    #con_string = pyodbc_connect_string
    #engine =Sqlalchemy.create_engine(con_string)
    #df = pd.read_sql(table, con=engine)


    with lock:
        df = pd.DataFrame([item.dict() for item in input_data])
        expected_types = get_type_hints(HouseInput)
        
        df = df.astype(expected_types)
        df = df.merge(demographics, on='zipcode', how='left')
        
        input_features = df[features]
        preds = model.predict(input_features)
    
    return {"prediction": [pred for pred in preds], "metadata": {"model_version": redis_client.get('model_version'), 'input_data_type':'full', 'input_data_size':'multi'}}


@app.get("/health")
async def health_check():
    try:
        with lock:
            test_input = HouseInputMinimal(**{'bedrooms': 3.0, 'bathrooms': 2.0, 'sqft_living': 1500.0, 'sqft_lot': 5000.0, 'floors': 1.0, 'sqft_above': 1500.0, 'sqft_basement': 0.0, 'zipcode': '98042'})
            df = pd.DataFrame([test_input.dict()])
            df = df.astype(get_type_hints(HouseInputMinimal))
            df = df.merge(demographics, on='zipcode', how='left')
            input_features = df[features]
            model.predict(input_features)
        return {"status": "healthy", "model_version": redis_client.get('model_version'), 'model_latest_update': redis_client.get('model_latest_update')}
    except Exception:
        return {"status": "unhealthy"}, 503
    
@app.post("/load_new_model")
async def load_new_model(new_model_name: str, new_model_version: str):
    global model
    
    new_model_path = 'model/' + new_model_name 
    if not os.path.exists(new_model_path):
        raise HTTPException(status_code=404, detail="New model file not found")
    
    with lock:
        try:
            new_model = pickle.load(open(new_model_path, 'rb'))
            
            if len(set(model.feature_names_in_) - set(new_model.feature_names_in_)) >0:
                raise HTTPException(status_code=500, detail="Failed to load model: inconsistent features")

            redis_client.set(MODEL_KEY, pickle.dumps(new_model))
            redis_client.set('model_version', new_model_version)
            redis_client.set('model_latest_update', str(datetime.datetime.now()))

            redis_client.publish("model_update", "new_model_loaded")
            model = new_model  

            return {"status": "success", "message": f"Model with name {new_model_name} updated and broadcasted"}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")