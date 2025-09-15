"""
This file contains the implementation of the random forest classifier model.
"""

import numpy as np
import optuna
import pandas as pd
import warnings
from copy import deepcopy
from MED3pa.models.abstract_models import ClassificationModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import Callable, Tuple


class RandomForestOptunaClassifier(ClassificationModel):
    """
    Random forest classifier model that is compatible with MED3pa requirements.
    """
    def __init__(self, objective: str = 'binary:logistic', class_weighting: bool = False, random_state: int = None,
                 verbose: bool = False, **params):
        """
        Initilizes the random forest classifier model instance.
        Args:
            objective: The objective of the model. Defaults to 'binary:logistic'.
            class_weighting: Whether to use class weighting for model training. Defaults to False.
            random_state: The random state to use. Defaults to None.
            verbose: Whether to print progress. Defaults to False.
            **params: Additional parameters specific to the sklearn random forest classifier model.
        """
        super().__init__(objective=objective, class_weighting=class_weighting, random_state=random_state,
                         verbose=verbose)
        self.model_class = RandomForestClassifier
        self.set_params(**params)
        self.pickled_model = True

    def fit(self, data: pd.DataFrame, target: np.ndarray, n_trials: int = 100, timeout: int = None,
            threshold: str | None = None, calibrate: bool = False, training_parameters: dict | None = None,
            balance_train_classes: bool | None = None, weights: np.ndarray | None = None):
        """
        Fits the random forest classifier instance.
        Args:
            data: Dataframe used to train the model.
            target: Targets to predict.
            n_trials: Number of training trials to perform with optuna. Defaults to 100.
            timeout: Timeout parameter for optuna. Stop study after the given number of second(s). Defaults to None.
            threshold: Threshold correction to apply to the model. Options are ['auc', 'auprc', None]. Defaults to None.
            calibrate: Wether to apply calibration to the model. Defaults to False.
            training_parameters: Dictionary of training parameters. Defaults to None.
            balance_train_classes: Whether to use class weighting for model training. Defaults to None.
            weights: Weights to apply to each sample. Defaults to None.

        """

        if balance_train_classes is not None:
            self._class_weighting = balance_train_classes

        if training_parameters:
            if self.params is not None:
                self.params.update(training_parameters)
            else:
                self.set_params(training_parameters)
        else:
            self.set_params()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if 'seed' in self.params:
            self._random_state = self.params['seed']
        elif 'random_state' in self.params:
            self._random_state = self.params['random_state']
        if self._random_state is not None:
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=self._random_state))
        else:
            study = optuna.create_study(direction="maximize")

        if calibrate:
            data, target, calibration_data, calibration_target = self._split_calibration_data(data, target)

        study.optimize(self._objective_fct(data, target), n_trials=n_trials, timeout=timeout)
        if len(study.trials) == 0 or all(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials):
            params = {'random_state': self._random_state}
        else:
            best_trial = study.best_trial
            params = best_trial.params
        self.params.update(params)
        self.model = RandomForestClassifier(**params)

        self.model.fit(data, target, sample_weight=weights)

        if calibrate:
            self.calibrate_model(y_true=calibration_target,
                                 data=calibration_data)

        if threshold:
            self._set_optimal_threshold(data, target, threshold)

        return self

    def _split_calibration_data(self, data: pd.DataFrame, target: np.ndarray, frac: float = 0.3
                                ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """
        Splits the training data into training and calibration data.
        Args:
            data: Dataframe train data to be split.
            target: Target variable to be predicted.
            frac: Proportion of training data to be used as calibration.

        Returns:
            data: Dataframe train data.
            target: Target data for training.
            calibration_data: Dataframe to calibrate the model.
            calibration_target: Target variable for the calibration of the model.

        """
        data = deepcopy(data)
        target = deepcopy(target)
        data['target'] = target
        calibration_data = data.sample(frac=frac, random_state=self._random_state)
        data = data.drop(calibration_data.index)
        calibration_target = calibration_data['target']
        calibration_data = calibration_data.drop(columns=['target'])
        target = data['target']
        data = data.drop(columns=['target'])
        return data, target, calibration_data, calibration_target

    def _set_optimal_threshold(self, data: pd.DataFrame, target: np.ndarray, threshold: str) -> None:
        """
        Sets the optimal classification threshold for the given threshold strategy
        Args:
            data: Data used for threshold optimization
            target: Target to be predicted
            threshold: Chosen threshold strategy, options: ['auc', 'auprc']

        """
        predicted = self.predict_proba(data)[:, 1]
        if threshold.lower() == 'auc':
            self._threshold = self._optimal_threshold_auc(target=target, predicted=predicted)
        elif threshold.lower() == 'auprc':
            self._threshold = self._optimal_threshold_auprc(target=target, predicted=predicted)
        else:
            raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the probability of each class.
        Args:
            X: Data to be predicted.

        Returns:
            Probabilities of each class

        """
        if self._calibration:
            probability = self._calibration.predict_proba(X)
        else:
            probability = self.model.predict_proba(X)
        return probability

    def _objective_fct(self, data: pd.DataFrame, target: np.ndarray) -> Callable[[optuna.trial], float]:
        """
        Objective function for optuna trials.
        Args:
            data: Data to optimize the model with optuna.
            target: Target variable to optimize the model with optuna.

        Returns:
            The objective function to train the model with optuna.

        """
        params_global = self.get_params()
        def __objective(trial):
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
            }

            if self._class_weighting:
                param["class_weight"] = "balanced"

            # if params is not None:
            #     param.update(params)

            if 'seed' in params_global:
                param['random_state'] = params_global['seed']
            elif 'random_state' in param:
                param['random_state'] = params_global['random_state']
            else:
                param['random_state'] = self._random_state

            clf = RandomForestClassifier(**param)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auc = np.mean(cross_val_score(clf, data, target, cv=5, scoring='roc_auc'))

            if np.isnan(auc):
                return 0
            return auc

        return __objective
