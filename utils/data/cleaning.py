"""
Data Cleaning Module
====================
Functions for loading and cleaning datasets.
"""

import pandas as pd
import numpy as np


def data_cleaning_food_dataset(dirpath):
    """
    Load and clean the food calories dataset.
    
    Args:
        dirpath: Path to the CSV file
    
    Returns:
        Cleaned DataFrame with standardized columns
    """
    data = pd.read_csv(dirpath, sep=';')
    data['makanan'] = data['makanan'] + ' ' + data['porsi']

    col_prep = data.columns[6:15].tolist()
    for col in col_prep:
        data[col] = data[col].fillna(0).astype(str)
        data[col] = np.where(
            data[col].str.contains('(mg)'),
            (data[col].str.replace(r'[a-zA-Z]', '', regex=True).astype(float) / 1000).astype(float),
            data[col].str.replace(r'[a-zA-Z]', '', regex=True).astype(float)
        )

    return data
