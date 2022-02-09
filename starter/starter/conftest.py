"""
[summary]

"""
import pandas as pd
import pytest 
import yaml
import logging


@pytest.fixture(scope='session')
def data():
    """[summary]

    Returns:
        [type]: [description]
    """
    with open('config.yaml','rb') as yml:
        config = yaml.safe_load(yml)
    
    data = pd.read_csv(config['main']['data_pth'])
    
    return data