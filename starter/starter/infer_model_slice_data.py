"""
Script to train machine learning model.
Author Hitoshi Kumagain
Data Feb 2022
""" 

import argparse
import yaml
import pickle
import logging
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model,compute_model_metrics,inference
from sklearn.model_selection import train_test_split

# Load yaml file to get config parameter
with open('config.yaml','rb') as yml:
    config = yaml.safe_load(yml)

# Set logging function default parameter
logging.basicConfig(
    filename=config['main']['log_pth'],
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Add code to load in the data.
data = pd.read_csv(config['main']['data_pth']) 
logging.info("Extract cleaned data by config parameter")



cat_features = config['model']['cat_features']
logging.info("Load category feature label from config parameter")

with open(config['main']["model_pth"],'rb') as mdl:
    model = pickle.load(open(config['main']["model_pth"],'rb'))
logging.info('Load inference model from model directory')

# todo モデル作成時の情報を読み込むように修正
train, test = train_test_split(data, test_size=config['model']['test_size'])
logging.info("Splitting data for test-train input")

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=config['model']['label'], training=True
)
logging.info('Encode data to create train dataset')
# todo

for feature in cat_features:
    for row_val in data[feature].unique():
        data_slice = data[data[feature]==row_val]
        if data_slice.shape[0]>1:
            # Optional enhancement, use K-fold cross validation instead of a train-test split.
            train, test = train_test_split(data_slice, test_size=config['model']['test_size'])
            logging.info("Splitting data for test-train input")

            # Process the test data with the process_data function.
            X_test, y_test, encoder, lb = process_data(
                test, categorical_features=cat_features,encoder=encoder, lb=lb,label=config['model']['label'], training=False
            )
            logging.info('Encode data to create test dataset')

            preds = inference(model,X_test)
            precision, recall, fbeta =compute_model_metrics(y_test, preds)
            logging.info('Make prefdiction and print model metrics')

            print(f"Column: {feature},Precision:{precision:.3f}, Recall:{recall:.3f}, F1 value:{fbeta:.3f}")
