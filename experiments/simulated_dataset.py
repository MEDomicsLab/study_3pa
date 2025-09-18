"""
Create experiments for the in-hospital mortality problem using MIMIC and eICU datasets.
"""

import pandas as pd
import pickle
import matplotlib
from MED3pa.datasets import DatasetsManager
from MED3pa.models import BaseModelManager
from MED3pa.med3pa import Med3paExperiment
from MED3pa.visualization.profiles_visualization import visualize_tree
from MED3pa.visualization.mdr_visualization import visualize_mdr


# Constants
params = {
    'calibrate': True,  # Whether to apply calibration correction to the BaseModel or not
    'class_weighting': True,  # Whether to apply class weighting correction in the BaseModel or not
    'main_seed': 42,
    'threshold': 'auc',  # Whether to apply threshold correction in the BaseModel or not. Options: ['auc', None]
    'to_saps_score': False  # True to convert variables to Saps scores, False to keep original variable value
}


def simulated_data_experiment():
    # import datasets
    train_df = pd.read_csv('datasets/simulated_dataset/simulated_train_data.csv')
    reference_df = pd.read_csv('datasets/simulated_dataset/simulated_reference_data.csv')
    test_df = pd.read_csv('datasets/simulated_dataset/simulated_test_data.csv')

    train_data = {'x': train_df[['x1', 'x2']], 'y': train_df['y_true'].to_numpy()}
    reference_data = {'x': reference_df[['x1', 'x2']], 'y': reference_df['y_true'].to_numpy()}
    test_data = {'x': test_df[['x1', 'x2']], 'y': test_df['y_true'].to_numpy()}

    # get BaseModel
    clf = pickle.load(open('datasets/simulated_dataset/clf.pkl', 'rb'))

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
    datasets.set_from_data(dataset_type="reference", observations=reference_data['x'].to_numpy(),
                           true_labels=reference_data['y'],
                           column_labels=reference_data['x'].columns)

    # Set testing Data for MED3pa evaluation
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
    results.save(file_path=f'experiments/results/simulated_dataset')
    visualize_mdr(result=results, filename='experiments/results/simulated_dataset/mdr')
    visualize_tree(result=results, filename='experiments/results/simulated_dataset/profiles')


if __name__ == '__main__':
    simulated_data_experiment()
