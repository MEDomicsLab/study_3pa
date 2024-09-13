"""
This file contains code to process datasets for the In-Hospital Mortality study
"""
from typing import Optional

import pandas as pd


def round_binary_variables(df: pd.DataFrame) -> pd.DataFrame:
    variables_to_round = ['mets', 'hem', 'aids', 'cpap', 'vent']
    df[variables_to_round] = df[variables_to_round].round()


def ventilation_correction(row):
    """
    Replaces the null values of cpap and vent with 0 when the patient has no pao2fio2 values, or with 1 otherwise
    :param row: the row (patient) on which to apply the ventilation correction
    :return: The correction applied to the row
    """
    row_isnull = row.isnull()
    if row_isnull['cpap'] and row_isnull['vent']:
        if row_isnull['pao2fio2']:
            row['vent'] = row['cpap'] = 0
        else:
            row['cpap'] = row['vent'] = 1
    return row
