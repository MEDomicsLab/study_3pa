import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ioff()  # Turn off interactive plotting
import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import roc_curve, precision_recall_curve, balanced_accuracy_score, recall_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef, f1_score, precision_score, RocCurveDisplay


class BaseModel(BaseEstimator, ClassifierMixin):
    clf = None
    THRESHOLD = 0.5
    calibration = None
    random_state = None
    classes_ = np.array([0, 1])  # To allow model calibration. The CalibratedClassifierCV uses this variable to
    # ensure that the model to be calibrated has been fitted.

    def __init__(self):
        raise NotImplementedError

    def calibrate_model(self, y_pred, y_true, data=None, method='sklearn'):
        if method == 'sklearn':
            calibration = CalibratedClassifierCV(estimator=deepcopy(self), method='sigmoid', cv='prefit')
            calibration.fit(data, y_true)
        # elif method == 'spline':
        #     calibration = mli.SplineCalib(random_state=self.random_state)
        #     calibration.fit(y_model=y_pred, y_true=y_true)
        else:
            raise NotImplementedError
        self.calibration = calibration

    def fit(self):
        raise NotImplementedError

    def get_results(self, data, target):
        predictions = self.predict(data)
        probabilities = self.predict_proba(data)[:, 1]
        if len(np.unique(target)) != 2:  # If only one class
            return {}
        results = {'auc': roc_auc_score(y_true=target, y_score=probabilities),
                   'auprc': average_precision_score(y_true=target, y_score=probabilities),
                   'mcc': matthews_corrcoef(y_true=target, y_pred=predictions),
                   'f1_score': f1_score(y_true=target, y_pred=predictions, zero_division=0),
                   'sensitivity': recall_score(y_true=target, y_pred=predictions, pos_label=1, zero_division=0),
                   'specificity': recall_score(y_true=target, y_pred=predictions, pos_label=0, zero_division=0),
                   'balanced_accuracy': balanced_accuracy_score(y_true=target, y_pred=predictions),
                   'ppv': precision_score(y_true=target, y_pred=predictions, pos_label=1, zero_division=0)
                   }
        return results

    def plot_probability_distribution(self, X, y, save_path=None):
        """
        Plot the predicted probability distributions for each class in a binary classification model.

        Parameters:
        model : sklearn or similar binary classification model
            The trained binary classification model.
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The true labels.

        Returns:
        None
        """
        plt.clf()
        # Predict probabilities for each class
        probas = self.predict_proba(X)[:, 1]

        # Separate probabilities for each class
        class_0_probas = probas[y == 0]
        class_1_probas = probas[y == 1]

        # Plot the probability distributions
        # plt.figure()
        plt.hist(class_1_probas, bins=20, alpha=0.5, label='Class 1', color='blue')  # , density=1)
        plt.hist(class_0_probas, bins=20, alpha=0.5, label='Class 0', color='red')  # , density=1)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency (%)')
        plt.title('Real Class Probability Distribution for Each Class')
        plt.legend(loc='upper center')
        # plt.gca().set_yticklabels(
        #     ['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])  # Format y-axis labels as percentages
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_roc_curve(self, X, target, save_path=None):
        plt.clf()
        predictions = self.predict_proba(X)[:, 1]
        fpr, tpr, threshold = roc_curve(target, predictions)
        roc_auc = roc_auc_score(target, predictions)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name=type(self).__name__)
        display.plot()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def predict_proba(self, data):
        raise NotImplementedError

    def predict(self, data):
        return (self.predict_proba(data)[:, 1] >= self.THRESHOLD).astype(int)

    def show_calibration(self, data, target, save_path=None):
        plt.clf()
        predicted_prob = self.predict_proba(data)[:, 1]
        CalibrationDisplay.from_predictions(target, predicted_prob)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def _optimal_threshold_auc(target, predicted):
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf).abs().argsort()[:1]]

        return list(roc_t['threshold'])[0]

    @staticmethod
    def _optimal_threshold_auprc(target, predicted):
        precision, recall, threshold = precision_recall_curve(target, predicted)
        # Remove last element
        precision = precision[:-1]
        recall = recall[:-1]

        i = np.arange(len(recall))
        roc = pd.DataFrame({'tf': pd.Series(precision * recall, index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf).abs().argsort()[:1]]

        return list(roc_t['threshold'])[0]
