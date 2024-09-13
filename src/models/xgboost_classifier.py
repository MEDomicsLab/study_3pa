import matplotlib
import numpy as np
import optuna
import xgboost as xgb

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ioff()  # Turn off interactive plotting
from copy import deepcopy

from src.models.base_model import BaseModel


class XGBClassifier(BaseModel):

    def __init__(self, objective='binary:logistic', random_state=None, class_weighting=False):
        self.objective = objective
        self.random_state = random_state
        self.class_weighting = class_weighting

    def fit(self, data, target, n_trials=100, timeout: int = None, threshold: str = None, calibrate: bool = False):

        if any('_miss' in col for col in data.columns):
            data = deepcopy(data)
            data = data.loc[:, ~data.columns.str.contains('_miss')]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if self.random_state:
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=self.random_state))
        else:
            study = optuna.create_study(direction="maximize")

        if calibrate:  # Temporary, for comparable results between calibrated and non-calibrated experiments, otherwise for calibration
            data = deepcopy(data)
            target = deepcopy(target)
            data['target'] = target
            calibration_data = data.sample(frac=0.3, random_state=self.random_state)
            data = data.drop(calibration_data.index)
            calibration_target = calibration_data['target']
            calibration_data = calibration_data.drop(columns=['target'])
            target = data['target']
            data = data.drop(columns=['target'])

        study.optimize(self._objective(data, target), n_trials=n_trials, timeout=timeout)
        best_trial = study.best_trial
        params = best_trial.params
        params['objective'] = self.objective
        self.clf = xgb.XGBClassifier(**params)

        self.clf.fit(data, target)

        if calibrate:
            # print(f"Results before calibration: \n {self.get_results(calibration_data, calibration_target)}")
            # self.plot_probability_distribution(calibration_data, calibration_target, save_path="/home/local/USHERBROOKE/lefo2801/3pa_test_oym/3pa_upd/3PA/hosp/240308/cal_pre.png")
            # self.plot_roc_curve(calibration_data, calibration_target, save_path="/home/local/USHERBROOKE/lefo2801/3pa_test_oym/3pa_upd/3PA/hosp/240308/auc_pre.png")
            if type(calibration_data) is xgb.DMatrix:
                calibration_data = calibration_data.get_data().toarray()
            self.calibrate_model(y_pred=self.clf.predict_proba(calibration_data)[:, 1],
                                 y_true=calibration_target,
                                 data=calibration_data)
            # print(f"Results After calibration: \n {self.get_results(calibration_data, calibration_target)}")
            # self.plot_probability_distribution(calibration_data, calibration_target, save_path="/home/local/USHERBROOKE/lefo2801/3pa_test_oym/3pa_upd/3PA/hosp/240308/cal_post.png")
            # self.plot_roc_curve(calibration_data, calibration_target, save_path="/home/local/USHERBROOKE/lefo2801/3pa_test_oym/3pa_upd/3PA/hosp/240308/auc_post.png")

        if threshold:
            predicted = self.predict_proba(data)[:, 1]
            if threshold.lower() == 'auc':
                self.THRESHOLD = self._optimal_threshold_auc(target=target, predicted=predicted)
            elif threshold.lower() == 'auprc':
                self.THRESHOLD = self._optimal_threshold_auprc(target=target, predicted=predicted)
            else:
                raise NotImplementedError

    def predict_proba(self, X):
        # temporary for a test
        if ['oh_insurance_Medicaid'] in X.columns.values:
            X = deepcopy(X)
            X = X.drop(columns=['oh_insurance_Medicaid', 'oh_insurance_Medicare', 'oh_insurance_Other',
                                'oh_language_ENGLISH', 'oh_ethnicity_AMERICAN INDIAN/ALASKA NATIVE',
                                'oh_ethnicity_OTHER', 'oh_ethnicity_ASIAN', 'oh_ethnicity_BLACK/AFRICAN AMERICAN',
                                'oh_ethnicity_HISPANIC/LATINO', 'oh_ethnicity_UNABLE TO OBTAIN',
                                'oh_ethnicity_UNKNOWN', 'oh_ethnicity_WHITE', 'oh_gender_M'])
        if any('_miss' in col for col in X.columns):
            X = deepcopy(X)
            X = X.loc[:, ~X.columns.str.contains('_miss')]

        if self.calibration:
            if type(X) is xgb.DMatrix:
                probability = self.calibration.predict_proba(X.get_data().toarray())
            else:
                probability = self.calibration.predict_proba(X)
        else:
            probability = self.clf.predict_proba(X)
        return probability

    def _objective(self, data, target):
        dtrain = xgb.DMatrix(data, label=target)

        def __objective(trial):
            param = {
                "device": "cpu",
                "verbosity": 0,
                "objective": self.objective,
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

            param['random_state'] = self.random_state

            if self.class_weighting:
                param['scale_pos_weight'] = len(target[target == 0]) / len(target[target == 1])
            #     samples_weights = class_weight.compute_sample_weight(
            #         class_weight='balanced',
            #         y=target
            #     )
            # else:
            #     samples_weights = None

            metric = np.mean(xgb.cv(params=param, dtrain=dtrain, nfold=5, metrics='auc')['test-auc-mean'])
            # metric = np.mean(cross_val_score(xgb.XGBClassifier(random_state=self.random_state, **param), data,
            #                                  target, scoring='roc_auc',
            #                                  fit_params={'sample_weight': samples_weights}))
            return metric

        return __objective
