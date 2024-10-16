"""
Create experiments for the in-hospital mortality problem using MIMIC and eICU datasets
"""

import pandas as pd

from MED3pa.datasets import DatasetsManager
from MED3pa.models import BaseModelManager
from MED3pa.med3pa import Med3paExperiment, Med3paDetectronExperiment
# from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

from src.models.random_forest_classifier import RandomForestOptunaClassifier
from datasets.POYM.constants import get_predictors

from sklearn.model_selection import train_test_split

# Constants
params = {
    'main_seed': 42
}


def poym_experiment():
    # import datasets
    df_train = pd.read_csv('datasets/POYM/df_train.csv')
    df_holdout = pd.read_csv('datasets/POYM/df_holdout.csv')

    # get predictors
    predictors, target = get_predictors(df_train)

    # Get reference set from training set
    patient_ids = df_train['Patient ID'].unique()
    # 70% train, 30% reference set
    train_ids, reference_ids = train_test_split(patient_ids, test_size=0.3, random_state=params['main_seed'])
    df_reference = df_train[df_train['Patient ID'].isin(reference_ids)].reset_index(drop=True)
    df_train = df_train[df_train['Patient ID'].isin(train_ids)].reset_index(drop=True)

    train_data = {'x': df_train[predictors],
                  'y': df_train[target]}

    test_data = {'x': df_holdout[predictors],
                 'y': df_holdout[target]}

    reference_data = {'x': df_reference[predictors],
                      'y': df_reference['oym']}

    # get BaseModel
    clf = RandomForestOptunaClassifier(random_state=params['main_seed']).fit(train_data['x'], train_data['y'])

    # ## MED3pa section
    # Set the base model using BaseModelManager
    base_model_manager = BaseModelManager()
    base_model_manager.set_base_model(model=clf)

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
    apc_params = {'max_depth': 3}

    # Initialize the DatasetsManager
    datasets = DatasetsManager()

    datasets.set_from_data(dataset_type="training", observations=train_data['x'].to_numpy(),
                           true_labels=train_data['y'],
                           column_labels=train_data['x'].columns)

    datasets.set_from_data(dataset_type="reference", observations=reference_data['x'].to_numpy(),
                           true_labels=reference_data['y'],
                           column_labels=reference_data['x'].columns)

    datasets.set_from_data(dataset_type="testing", observations=test_data['x'].to_numpy(),
                           true_labels=test_data['y'],
                           column_labels=test_data['x'].columns)

    # Execute the MED3PA experiment
    results = Med3paExperiment.run(
        datasets_manager=datasets,
        base_model_manager=base_model_manager,
        uncertainty_metric="sigmoidal_error",
        ipc_type='RandomForestRegressor',
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
    results.save(file_path=f'experiments/results/poym')

    # Execute the Med3pa experiment with Detectron results
    med3pa_detectron_results = Med3paDetectronExperiment.run(
        datasets=datasets,
        base_model_manager=base_model_manager,
        uncertainty_metric="sigmoidal_error",
        ipc_type='RandomForestRegressor',
        ipc_params=ipc_params,
        apc_params=apc_params,
        ipc_grid_params=ipc_grid,
        apc_grid_params=apc_grid,
        samples_size=20,
        ensemble_size=10,
        num_calibration_runs=100,
        patience=3,
        test_strategies="enhanced_disagreement_strategy",
        allow_margin=False,
        margin=0.05,
        samples_ratio_min=0,
        samples_ratio_max=10,
        samples_ratio_step=5,
        evaluate_models=True,
    )

    # Save the results to a specified directory
    med3pa_detectron_results.save(file_path=f'experiments/results/poym/with_detectron')


if __name__ == '__main__':
    poym_experiment()
