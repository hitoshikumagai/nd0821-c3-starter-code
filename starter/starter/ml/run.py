import pandas as pd
import numpy as np
import data

df = pd.read_csv('../../data/census.csv') 

def get_categorical_features(df:pd.DataFrame,label:str):
    feature_set=set(df.columns.values)
    num_set=set(df.describe().columns.values)
    categorical_features=list(feature_set-num_set)
    categorical_features = [item for item in categorical_features if item != label]  
    return categorical_features

if __name__ == '__main__':
    categorical_features=get_categorical_features(df,' salary')
    X,y,encoder,lb=data.process_data(df,categorical_features,' salary')
