"""
[summary]

"""
import pandas as pd
from pytest import fixture
import logging


logging.basicConfig(
    filename='../logs/census_train.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


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
    except AssertionError as error:
        logging.warning('Input data is not extracted.')
        raise error
