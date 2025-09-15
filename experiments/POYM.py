"""
Create experiments for the One-Year Mortality task using Data from https://doi.org/10.1007/s13755-024-00332-4.
"""

import pandas as pd
import pickle
from datetime import datetime
from MED3pa.datasets import DatasetsManager
from MED3pa.models import BaseModelManager
from MED3pa.med3pa import Med3paExperiment, Med3paDetectronExperiment
from MED3pa.visualization.profiles_visualization import visualize_tree
from MED3pa.visualization.mdr_visualization import visualize_mdr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.models.random_forest_classifier import RandomForestOptunaClassifier
from datasets.POYM.constants import get_predictors


# Constants
params = {
    'calibrate': True,  # Whether to apply calibration correction to the BaseModel or not
    'class_weighting': True,  # Whether to apply class weighting correction in the BaseModel or not
    'main_seed': 42,
    'threshold': 'auc',  # Whether to apply threshold correction in the BaseModel or not. Options: ['auc', None]
    'fit_baseModel': False
}


def poym_experiment():
    # import datasets
    df_train = pd.read_csv('datasets/POYM/df_train.csv')
    df_holdout = pd.read_csv('datasets/POYM/df_holdout.csv')

    # get predictors
    predictors, target, cat_cols = get_predictors(df_train)

    # Convert categorical variables
    encoder = OneHotEncoder(drop='if_binary').fit(df_train[cat_cols])
    encoded_array_train = encoder.transform(df_train[cat_cols]).toarray()
    encoded_array_holdout = encoder.transform(df_holdout[cat_cols]).toarray()

    # Create DataFrame with encoded variables
    encoded_df_train = pd.DataFrame(encoded_array_train, columns=encoder.get_feature_names_out(cat_cols))
    encoded_df_holdout = pd.DataFrame(encoded_array_holdout, columns=encoder.get_feature_names_out(cat_cols))

    # Drop original columns and concatenate
    df_train = df_train.drop(columns=cat_cols)
    df_holdout = df_holdout.drop(columns=cat_cols)

    df_train = pd.concat([df_train, encoded_df_train], axis=1)
    df_holdout = pd.concat([df_holdout, encoded_df_holdout], axis=1)

    predictors = list(filter(lambda x: x not in cat_cols, predictors))
    predictors += list(encoder.get_feature_names_out(cat_cols))

    # Get reference set from training set
    patient_ids = df_train['patient_id'].unique()
    # 70% train, 30% reference set
    train_ids, reference_ids = train_test_split(patient_ids, test_size=0.3, random_state=params['main_seed'])
    df_reference = df_train[df_train['patient_id'].isin(reference_ids)].reset_index(drop=True)
    df_train = df_train[df_train['patient_id'].isin(train_ids)].reset_index(drop=True)

    train_data = {'x': df_train[predictors],
                  'y': df_train[target].to_numpy(),
                  'columns': predictors}

    test_data = {'x': df_holdout[predictors].to_numpy(),
                 'y': df_holdout[target].to_numpy(),
                 'columns': predictors}

    reference_data = {'x': df_reference[predictors].to_numpy(),
                      'y': df_reference[target].to_numpy(),
                      'columns': predictors}

    # get BaseModel
    if not params['fit_baseModel']:
        try:
            clf = pickle.load(open('datasets/POYM/clf.pkl', 'rb'))
        except FileNotFoundError:
            clf = None

    if params['fit_baseModel'] or clf is None:
        print(f"Starting BaseModel training :{datetime.now()}")
        clf = RandomForestOptunaClassifier(random_state=params['main_seed'],
                                           class_weighting=params['class_weighting']
                                           ).fit(train_data['x'],
                                                 train_data['y'],
                                                 calibrate=params['calibrate'],
                                                 threshold=params['threshold'])
        pickle.dump(clf, open('datasets/POYM/clf.pkl', 'wb'))

    # ## MED3pa section
    # Set the base model using BaseModelManager
    base_model_manager = BaseModelManager(model=clf)

    # Define parameters for the experiment
    ipc_params = {'n_estimators': 100}
    ipc_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 4]
    }
    apc_grid = {
        'max_depth': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 4]
    }
    apc_params = {'max_depth': 6}

    # Initialize the DatasetsManager
    datasets = DatasetsManager()

    # Set reference Data (Same distribution as training data)
    datasets.set_from_data(dataset_type="reference", observations=reference_data['x'],
                           true_labels=reference_data['y'],
                           column_labels=reference_data['columns'])

    # Set testing Data for MED3pa evaluation
    datasets.set_from_data(dataset_type="testing", observations=test_data['x'],
                           true_labels=test_data['y'],
                           column_labels=test_data['columns'])

    # Execute the MED3PA experiment
    print(f"Starting Med3PA experiment :{datetime.now()}")
    med3pa_results = Med3paExperiment.run(
        datasets_manager=datasets,
        base_model_manager=base_model_manager,
        uncertainty_metric="sigmoidal_error",
        ipc_type='EnsembleRandomForestRegressor',
        ipc_params=ipc_params,
        apc_params=apc_params,
        ipc_grid_params=ipc_grid,
        apc_grid_params=apc_grid,
        samples_ratio_min=0,
        samples_ratio_max=10,
        samples_ratio_step=5,
        evaluate_models=True,
    )

    # Save the results to a specified directory
    med3pa_results.save(file_path=f'experiments/results/poym')
    visualize_mdr(result=med3pa_results, filename='experiments/results/poym/mdr')
    visualize_tree(result=med3pa_results, filename='experiments/results/poym/profiles')


if __name__ == '__main__':
    poym_experiment()
