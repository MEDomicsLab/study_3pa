"""
This file contains code to process datasets for the In-Hospital Mortality study.
"""

import pandas as pd


def ventilation_correction(row: pd.Series) -> pd.Series:
    """
    Replaces the null values of cpap and vent with 0 when the patient has no pao2fio2 values, or with 1 otherwise.

    Args:
        row (pd.Series): The row (patient) on which to apply the ventilation correction.
    """
    row_isnull = row.isnull()
    if row_isnull['cpap'] and row_isnull['vent']:
        if row_isnull['pao2fio2']:
            row['vent'] = row['cpap'] = 0
        else:
            row['cpap'] = row['vent'] = 1
    return row
