"""
Master Thesis
Academic year 2021-2022

Authors:
    - GILSON Maxence
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from used_metrics import get_directory
import matplotlib.pyplot as plt
from database import DB

import numpy as np


def get_selecting_features(Classifier, train_DB, train_labels, n_features):
    selection_RFE = RFE(Classifier, n_features_to_select=n_features, step=1)
    selection_RFE.fit(train_DB, train_labels)
    selected_feature = train_DB.columns[(selection_RFE.get_support())]
    return selected_feature


def plot_feature_importance(Classifier, train_DB, train_labels, method):
    Classifier.fit(train_DB, train_labels)
    imp = Classifier.feature_importances_
    # Plotting feature importance scores
    directory = get_directory(method)
    indices = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(26, 18))
    ax.barh(range(len(imp)), imp[indices])
    ax.set_yticks(range(len(imp)))
    _ = ax.set_yticklabels(np.array(train_DB.columns)[indices])
    plt.ylabel('Feature importance score')
    plt.xlabel('Features')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(directory + "Feature Importance")
    return


def get_scores(test_labels, prediction):
    # Score is scoring the testing data
    acc_score = accuracy_score(test_labels, prediction)
    f_score = f1_score(test_labels, prediction, average='macro')
    return acc_score, f_score


def sup_method(method, labels, test_size, feature_selection, n_features, efficiency):
    # Creating training and testing datasets
    train_DB, test_DB, train_labels, test_labels = train_test_split(DB, labels, test_size=test_size, train_size=None)
    if method == "RF":
        classifier = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None, max_features='sqrt',
                                            bootstrap=True)
    elif method == "SVM":
        classifier = LinearSVC(C=1.0)
    else:
        return "Wrong supervised method"
    # Fit is for training data
    fitted = classifier.fit(train_DB, train_labels)
    prediction = fitted.predict(test_DB)
    if feature_selection:
        if method == "RF":
            plot_feature_importance(classifier, train_DB, train_labels, method)
        return get_selecting_features(classifier, train_DB, train_labels, n_features)
    if efficiency:
        return get_scores(test_labels, prediction)
    return prediction
