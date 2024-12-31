from typing import Any, Dict, Optional, List

import matplotlib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
optuna.logging.set_verbosity(optuna.logging.ERROR)

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ioff()  # Turn off interactive plotting
from copy import deepcopy

from MED3pa.models.abstract_models import ClassificationModel


class XGBClassifier(ClassificationModel):
    def __init__(self, objective: str = 'binary:logistic', class_weighting: bool = False, random_state: int = None):
        super().__init__(objective=objective, class_weighting=class_weighting, random_state=random_state)
        self.model_class = xgb.Booster

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

        if calibrate:  # Temporary, for comparable results between calibrated and non-calibrated experiments, otherwise for calibration
            data = deepcopy(data)
            target = deepcopy(target)
            data['target'] = target
            calibration_data = data.sample(frac=0.3, random_state=self._random_state)
            data = data.drop(calibration_data.index)
            calibration_target = calibration_data['target']
            calibration_data = calibration_data.drop(columns=['target'])
            target = data['target']
            data = data.drop(columns=['target'])

        study.optimize(self._objective_fct(data, target), n_trials=n_trials, timeout=timeout)
        if len(study.trials) == 0 or all(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials):
            params = {}
        else:
            best_trial = study.best_trial
            params = best_trial.params
        params['objective'] = self._objective
        self.set_params(params)
        self.model = xgb.XGBClassifier(**params)

        self.model.fit(data, target, sample_weight=weights)

        if calibrate:
            # print(f"Results before calibration: \n {self.get_results(calibration_data, calibration_target)}")
            # self.plot_probability_distribution(calibration_data, calibration_target, save_path="/home/local/USHERBROOKE/lefo2801/3pa_test_oym/3pa_upd/3PA/hosp/240308/cal_pre.png")
            # self.plot_roc_curve(calibration_data, calibration_target, save_path="/home/local/USHERBROOKE/lefo2801/3pa_test_oym/3pa_upd/3PA/hosp/240308/auc_pre.png")
            if type(calibration_data) is xgb.DMatrix:
                calibration_data = calibration_data.get_data().toarray()
            elif isinstance(calibration_data, pd.DataFrame):
                calibration_data = calibration_data.to_numpy()
            self.calibrate_model(y_pred=self.model.predict_proba(calibration_data)[:, 1],
                                 y_true=calibration_target,
                                 data=calibration_data)
            # print(f"Results After calibration: \n {self.get_results(calibration_data, calibration_target)}")
            # self.plot_probability_distribution(calibration_data, calibration_target, save_path="/home/local/USHERBROOKE/lefo2801/3pa_test_oym/3pa_upd/3PA/hosp/240308/cal_post.png")
            # self.plot_roc_curve(calibration_data, calibration_target, save_path="/home/local/USHERBROOKE/lefo2801/3pa_test_oym/3pa_upd/3PA/hosp/240308/auc_post.png")

        if threshold:
            predicted = self.predict_proba(data)[:, 1]
            if threshold.lower() == 'auc':
                self._threshold = self._optimal_threshold_auc(target=target, predicted=predicted)
            elif threshold.lower() == 'auprc':
                self._threshold = self._optimal_threshold_auprc(target=target, predicted=predicted)
            else:
                raise NotImplementedError("Threshold correction methods must be 'auc' or 'auprc', not "
                                          "{}".format(threshold))

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if self._calibration:
            if type(X) is xgb.DMatrix:
                probability = self._calibration.predict_proba(X.get_data().toarray())
            else:
                probability = self._calibration.predict_proba(X)
        else:
            probability = self.model.predict_proba(X)
        return probability

    def _objective_fct(self, data, target):
        dtrain = xgb.DMatrix(data, label=target)
        params = self.get_params()

        def __objective(trial):
            param = {
                "device": "cpu",
                "verbosity": 0,
                "objective": self._objective,
                # use exact for small dataset.
                "tree_method": "exact",
                # defines booster, gblinear for linear functions.
                "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                # L2 regularization weight.
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                # L1 regularization weight.
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                # sampling ratio for training data.
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                # sampling according to each tree.
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            }

            if param["booster"] in ["gbtree", "dart"]:
                # maximum depth of the tree, signifies complexity of the tree.
                param["max_depth"] = trial.suggest_int("max_depth", 3, 11, step=2)
                # minimum child weight, larger the term more conservative the tree.
                param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
                param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
                # defines how selective algorithm is.
                param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
                param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

            param['random_state'] = self._random_state

            if self._class_weighting:
                param['scale_pos_weight'] = len(target[target == 0]) / len(target[target == 1])
            #     samples_weights = class_weight.compute_sample_weight(
            #         class_weight='balanced',
            #         y=target
            #     )
            # else:
            #     samples_weights = None
            if params is not None:
                param.update(params)

            metric = np.mean(xgb.cv(params=param, dtrain=dtrain, nfold=5, metrics='auc')['test-auc-mean'])
            # metric = np.mean(cross_val_score(xgb.XGBClassifier(random_state=self._random_state, **param), data,
            #                                  target, scoring='roc_auc',
            #                                  fit_params={'sample_weight': samples_weights}))
            if np.isnan(metric):
                return 0
            return metric

        return __objective
