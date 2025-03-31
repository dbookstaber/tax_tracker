"""Test configuration for pytest"""
import sys
import os
import pytest
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope='session')
def distribution_data():
    """Fixture to read distribution data from the tests directory"""
    test_dir = os.path.dirname(__file__)
    dist_data_path = os.path.join(test_dir, 'DistributionData.csv')
    return pd.read_csv(dist_data_path, index_col=[0, 1, 2], parse_dates=[1]).sort_index()

@pytest.fixture(scope='session')
def price_data():
    """Fixture to read price data from the tests directory"""
    test_dir = os.path.dirname(__file__)
    price_data_path = os.path.join(test_dir, 'PriceData.csv')
    return pd.read_csv(price_data_path, index_col=[0, 1], parse_dates=[1]).sort_index()
