"""
[summary]
Author: Hitoshi Kumagai
Date: Feb. 2022
"""
import pandas as pd
import pytest
from pytest import fixture
import logging

with open('config.yaml','rb') as yml:
    config = yaml.safe_load(yml)

logging.basicConfig(
    filename='starter/logs/census_train.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture(scope='session')
def data():
    """[summary]
    Returns:
        [type]: [description]
    """
    data = pd.read_csv('starter/data/census_cleaned.csv')
    
    return data

def test_extract_data(data:pd.DataFrame):
    """[summary]
    Args:
        data ([pandas dataframe): Cleaned dataframe from csv file
    Raises:
        error: Number of column and row is not correctry extracted.
    """
    try:
        assert data.shape[0]==32561
        assert data.shape[1]==15
        logging.info('Cleaned data is extracted in dataframe:SUCCESS')
    except FileNotFoundError as error:
        logging.warning('Input data is not extracted.')
        raise error
