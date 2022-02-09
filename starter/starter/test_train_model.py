"""
[summary]

"""
import pandas as pd
import yaml

with open('config.yaml','rb') as yml:
    config = yaml.safe_load(yml)

data = pd.read_csv(config['main']['data_pth'])

def test_extract_data(data):
    try:
        assert data.shape[0]>0
        assert data.shape[1]>0
    except AssertionError as error:
        print('Error')
        raise error
