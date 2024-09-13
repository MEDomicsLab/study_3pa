"""
Create experiments for the in-hospital mortality problem using MIMIC and eICU data
"""

import pandas as pd

from src.data.processing_in_hospital_mortality import ventilation_correction

# Import datasets
mimic_df = pd.read_csv('datasets/in-hospital_mortality/mimic_filtered_data.csv')
eicu_df = pd.read_csv('datasets/in-hospital_mortality/eicu_filtered_data.csv')

# Apply ventilation correction
mimic_df[['pao2fio2', 'cpap', 'vent']] = mimic_df.apply(lambda row:
                                                        ventilation_correction(row[['pao2fio2', 'cpap', 'vent']]),
                                                        axis=1)
