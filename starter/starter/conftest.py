"""
[summary]

"""
import pandas as pd
import pytest 
import logging


@pytest.fixture(scope='session')
def data():
    """[summary]
    Returns:
        [type]: [description]
    """
    data = pd.read_csv('../data/census_cleaned.csv')
    
    return data