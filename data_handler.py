import pandas as pd
import random 
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
"""
This version of data_handler will just be using the kaggle, already-cleaned dataset so we can start work without
needing to wait for the data to be fully cleaned from the original UCI dataset.

Dataset heart.csv will only be used temporarily, attributes:
    Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
    University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
    University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
    V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
    Donor:
    David W. Aha (aha '@' ics.uci.edu) (714) 856-8779
"""


def get_data():
    df = pd.read_csv('heart.csv')
    desired_cols = [
        'age',
        'trestbps',
        'sex',
        'target'
    ]
    df = df[df.columns.intersection(desired_cols)]
    # Forming a feature/label array for ease of use with svm. Code is not nice but basically just converts to two arrays of the relevant features and labels needed for SVM classification
    features = [df[x].tolist() for x in desired_cols[:-1]]
    features = [normalise_data(x) for x in features]
    feat = [[features[i][j] for i in range(len(features))] for j in range(len(features[0]))]
    labels = df[desired_cols[-1]].tolist()
    feat_names = desired_cols[:-1]
    return feat_names, feat, labels


def normalise_data(data):
    mind = min(data)
    maxd = max(data)
    norm = [(x - mind)/(maxd-mind) for x in data]
    return norm


def split_data(features, labels):
    # The .csv data comes pre-sorted so we need to shuffle the data around
    features, labels = shuffle(features, labels)

    length = len(features)
    testing_ratio = 0.2

    # Separating the data into required arrays based on the ratio above
    training_features = features[int(length * testing_ratio):]
    training_labels = labels[int(length * testing_ratio):]
    test_features = features[:int(length * testing_ratio)]
    test_labels = labels[:int(length * testing_ratio)]

    return training_features, training_labels, test_features, test_labels


def k_fold_cross_validation(k, kernel, features, labels):
    #kernel specifies which kernel you want it's a string "linear","rfb" etc.
    #k is how many folds u want in the data we'll probably use 5 fold
    
    clf = svm.SVC(kernel=kernel, random_state=69)
    scores = cross_val_score(clf, features, labels, cv=k)
    return scores

# def get_heatmap():
#     df = pd.read_csv('heart.csv')
#     plt.figure(figsize=(12,10))
#     sns.heatmap(df.corr(), linewidths=0.05, fmt= ".2f", annot=True)
#     plt.title("Correlation Plot")
#     plt.show()
