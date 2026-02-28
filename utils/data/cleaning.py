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

    # Ensure column names are trimmed
    data.columns = [c.strip() for c in data.columns]

    # Combine makanan + porsi for display
    data['makanan'] = data.get('makanan', '').fillna('') + ' ' + data.get('porsi', '').fillna('')

    # Columns expected to hold numeric nutrition values (slice may vary by dataset)
    col_prep = data.columns[6:15].tolist()

    def parse_nutrient_value(val):
        try:
            if pd.isna(val):
                return 0.0
            s = str(val).strip()
            # Normalize decimal comma to dot
            s = s.replace(',', '.')
            # Detect milligram values (e.g. '391mg' or '391 mg' or '391mg)')
            lower = s.lower()
            is_mg = 'mg' in lower
            # Remove any non-numeric characters except dot and minus
            import re
            cleaned = re.sub(r'[^0-9.\-]', '', s)
            if cleaned == '' or cleaned == '.' or cleaned == '-':
                return 0.0
            num = float(cleaned)
            if is_mg:
                # convert mg to g for consistency (so sodium/kolesterol in mg become g)
                return num / 1000.0
            return num
        except Exception:
            return 0.0

    for col in col_prep:
        # Fill NA and convert using parse_nutrient_value
        data[col] = data[col].apply(parse_nutrient_value)

    return data
