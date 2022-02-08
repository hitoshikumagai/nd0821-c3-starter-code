"""

Author: 
Date: 
"""

# Script to train machine learning model.
import yaml
import pickle
import logging
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model,compute_model_metrics,inference
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.


with open('config.yaml','rb') as yml:
    config = yaml.safe_load(yml)

logging.basicConfig(
    filename=config['main']['log_pth'],
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Add code to load in the data.
data = pd.read_csv(config['main']['data_pth']) 
logging.info("Extract cleaned data by config parameter")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=config['model']['test_size'])
logging.info("Splitting data for test-train input")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=config['model']['label'], training=True
)
logging.info('Encode data to create train dataset')

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features,encoder=encoder, lb=lb,label=config['model']['label'], training=False
)
logging.info('Encode data to create test dataset')


# Train and save a model.
model = train_model(X_train, y_train)
logging.info('Create inference model by train data')

pickle.dump(model, open(config['main']["model_pth"], 'wb'))
logging.info('Save inference model to model directory')

preds = inference(model,X_test)
precision, recall, fbeta =compute_model_metrics(y_test, preds)
logging.info('Make prefdiction and print model metrics')

print(f"Precision:{precision:.3f}, Recall:{recall:.3f}, F1 value:{fbeta:.3f}")
