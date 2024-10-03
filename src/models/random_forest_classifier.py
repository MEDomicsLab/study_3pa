from typing import Any, Dict, Optional
import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from copy import deepcopy

from MED3pa.models.abstract_models import ClassificationModel


class RandomForestOptunaClassifier(ClassificationModel):
    def __init__(self, objective: str = 'binary:logistic', class_weighting: bool = False, random_state: int = None,
                 **params):
        super().__init__(objective=objective, class_weighting=class_weighting, random_state=random_state)
        self.model_class = RandomForestClassifier
        self.set_params(**params)

    def fit(self, data, target, n_trials=100, timeout: int = None, threshold: str = None, calibrate: bool = False,
            training_parameters: dict = None, balance_train_classes: bool = None, weights: np.ndarray = None):

        if balance_train_classes is not None:
            self._class_weighting = balance_train_classes

        if training_parameters:
            if self.params is not None:
                self.params.update(training_parameters)
            else:
                self.set_params(training_parameters)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if self._random_state:
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=self._random_state))
        else:
            study = optuna.create_study(direction="maximize")

        if calibrate:
            data, target, calibration_data, calibration_target = self._split_calibration_data(data, target)

        study.optimize(self._objective_fct(data, target), n_trials=n_trials, timeout=timeout)
        best_trial = study.best_trial
        params = best_trial.params
        self.set_params(params)
        self.model = RandomForestClassifier(**params, random_state=self._random_state)

        self.model.fit(data, target)

        if calibrate:
            self.calibrate_model(y_pred=self.model.predict_proba(calibration_data)[:, 1],
                                 y_true=calibration_target,
                                 data=calibration_data)

        if threshold:
            self._set_optimal_threshold(data, target, threshold)

        return self

    def _split_calibration_data(self, data, target, frac=0.3):
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

    def _set_optimal_threshold(self, data, target, threshold):
        predicted = self.predict_proba(data)[:, 1]
        if threshold.lower() == 'auc':
            self._threshold = self._optimal_threshold_auc(target=target, predicted=predicted)
        elif threshold.lower() == 'auprc':
            self._threshold = self._optimal_threshold_auprc(target=target, predicted=predicted)
        else:
            raise NotImplementedError

    def predict_proba(self, X):
        if self._calibration:
            probability = self._calibration.predict_proba(X)
        else:
            probability = self.model.predict_proba(X)
        return probability

    def _objective_fct(self, data, target):
        params = self.get_params()
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

            if params is not None:
                param.update(params)

            if 'seed' in param:
                random_state = param.pop('seed')
            elif 'random_state' in param:
                random_state = param.pop('random_state')
            else:
                random_state = self._random_state

            clf = RandomForestClassifier(**param, random_state=self._random_state)
            auc = np.mean(cross_val_score(clf, data, target, cv=5, scoring='roc_auc'))
            return auc

        return __objective
