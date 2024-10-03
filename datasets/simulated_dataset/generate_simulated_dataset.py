import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)

from src.models.random_forest_classifier import RandomForestOptunaClassifier


class DataGenerator:
    """
    Binary class Data generator
    """

    def __init__(self, generative_func=None):
        self.generative_func = generative_func

    def generate_data(self, n_data, means, stds, seed=None):
        np.random.seed(seed)
        data = []
        for N, mean, std in zip(n_data, means, stds):
            dist = np.random.randn(N, len(mean)) * std + mean
            data.append(dist)
        X_data = np.concatenate(data, axis=0)

        if self.generative_func is None:
            return X_data
        return X_data, self.generative_func(X_data)


# Function to create a meshgrid and predict over it
def plot_decision_regions(X, y, classifier, title, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('blue', 'orange')))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k', cmap=ListedColormap(('blue', 'orange')))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    ## Constants
    TRAIN_N = [680, 170]
    TRAIN_MEANS = [[4, 1.65], [4, -1.65]]
    TRAIN_STDS = [[3, 1.15], [3, 1.15]]

    TEST_N = [680, 170, 120, 30]
    TEST_MEANS = [[4, 2], [4, -2], [17, 0], [17, 0]]
    TEST_STDS = [[3, 1.15], [3, 1.15], [1.5, 2], [1.5, 2]]

    dataGenerator = DataGenerator()

    X_train = dataGenerator.generate_data(n_data=TRAIN_N, means=TRAIN_MEANS, stds=TRAIN_STDS, seed=19546)
    Y_train = np.append(np.zeros(TRAIN_N[0]), np.ones(TRAIN_N[1]))

    X_reference = dataGenerator.generate_data(n_data=TRAIN_N, means=TRAIN_MEANS, stds=TRAIN_STDS, seed=19547)
    Y_reference = np.append(np.zeros(TRAIN_N[0]), np.ones(TRAIN_N[1]))

    X_test = dataGenerator.generate_data(n_data=TEST_N, means=TEST_MEANS, stds=TEST_STDS, seed=19550)
    Y_test = np.concatenate([np.zeros(TEST_N[0]), np.ones(TEST_N[1]), np.zeros(TEST_N[2]), np.ones(TEST_N[3])])

    # Train classifier
    clf = RandomForestOptunaClassifier(max_depth=2, n_estimators=100, random_state=0).fit(X_train, Y_train)
    pickle.dump(clf, open('datasets/simulated_dataset/clf.pkl', 'wb'))

    # Create dataframes for easy plotting and saving
    df_train = pd.DataFrame(X_train, columns=['x1', 'x2'])
    df_train['y_true'] = Y_train

    df_reference = pd.DataFrame(X_reference, columns=['x1', 'x2'])
    df_reference['y_true'] = Y_reference
    df_reference.to_csv('datasets/simulated_dataset/simulated_reference_data.csv', index=False)

    df_test = pd.DataFrame(X_test, columns=['x1', 'x2'])
    df_test['y_true'] = Y_test

    # Save training set to CSV
    df_train.to_csv('datasets/simulated_dataset/simulated_train_data.csv', index=False)

    # Plot decision regions for training set
    plot_decision_regions(X_train, Y_train, clf, title="Training Set with Decision Regions",
                          filename='training_decision_regions.png')

    # Plot decision regions for test set
    plot_decision_regions(X_test, Y_test, clf, title="Test Set with Decision Regions",
                          filename='test_decision_regions.png')

    # Optionally save the test data
    df_test.to_csv('datasets/simulated_dataset/simulated_test_data.csv', index=False)

    # Predictions and probabilities for training and test sets
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    train_probs = clf.predict_proba(X_train)[:, 1]  # Probability of the positive class
    test_probs = clf.predict_proba(X_test)[:, 1]    # Probability of the positive class

    # Classification results for training set
    print("Training Set Results:")
    print(f"Accuracy: {accuracy_score(Y_train, train_preds):.4f}")
    print(f"AUC: {roc_auc_score(Y_train, train_probs):.4f}")
    print(f"AUPRC: {average_precision_score(Y_train, train_probs):.4f}")
    print("Classification Report:\n", classification_report(Y_train, train_preds))
    print("Confusion Matrix:\n", confusion_matrix(Y_train, train_preds))

    # Classification results for test set
    print("\nTest Set Results:")
    print(f"Accuracy: {accuracy_score(Y_test, test_preds):.4f}")
    print(f"AUC: {roc_auc_score(Y_test, test_probs):.4f}")
    print(f"AUPRC: {average_precision_score(Y_test, test_probs):.4f}")
    print("Classification Report:\n", classification_report(Y_test, test_preds))
    print("Confusion Matrix:\n", confusion_matrix(Y_test, test_preds))
