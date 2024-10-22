"""
Filename: constants.py

Authors: Hakima Laribi, Olivier Lefebvre

Description: Defines constants related to the dataset. Taken and adapted from the POYM project:
    https://github.com/MEDomics-UdeS/POYM/blob/main/src/data/processing/constants.py
"""

import re
import pandas as pd

from typing import Tuple, List


def get_predictors(df: pd.DataFrame) -> Tuple[List[str], str, List[str]]:
    """

    """

    # Comorbidities diagnostic variables
    DX_COLS = [dx for dx in list(df.columns) if re.compile("^(dx_).+").search(dx)]

    # Admission diagnosis variables
    ADM_COLS = [adm for adm in list(df.columns) if re.compile("^(adm_).+").search(adm)]

    # Demographic, previous care utilization and characteristics of the current admission variables
    OTHER_COLS = [
        "age_original",
        "gender",
        "ed_visit_count",
        "ho_ambulance_count",
        "flu_season",
        "living_status",
        "total_duration",
        "admission_group",
        "is_ambulance",
        "is_icu_start_ho",
        "is_urg_readm",
        "service_group",
        "has_dx"
    ]
    # Target variable
    OYM = "oym"

    # 244 PREDICTORS of HOMR
    PREDICTORS = OTHER_COLS + DX_COLS + ADM_COLS

    CONT_COLS = [
        "age_original",
        "ed_visit_count",
        "ho_ambulance_count",
        "total_duration"
    ]

    # Categorical variables
    initial_CAT_COLS = [cat for cat in DX_COLS + ADM_COLS + OTHER_COLS if cat not in CONT_COLS]

    CAT_COL = [cat for cat in initial_CAT_COLS if cat not in DX_COLS + ADM_COLS + ["has_dx"]]

    return PREDICTORS, OYM, ['gender', 'living_status', 'admission_group', 'service_group']
