import requests
import pandas as pd
import random 
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
"""
Dataset attributes:
    Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
    University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
    University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
    V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
    Donor:
    David W. Aha (aha '@' ics.uci.edu) (714) 856-8779
"""


def pull_data_from_db():
    """
    Pulls data from online UCI databases and collates them
    """
    base_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/'
    # Pulling processed data as it is in the most usable format but still needs to be cleaned (missing data)
    options = [
        'processed.cleveland.data',
        'processed.hungarian.data',
        'processed.switzerland.data',
        'processed.va.data',
    ]
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalanch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]

    # This chunk parses the online data into float arrays where it can and adds it to all_data together
    all_data = []
    for o in options:
        req = requests.get(base_data_url + o)
        if req.status_code != 200:
            raise requests.exceptions.ConnectionError(f'Data URL {base_data_url + o} invalid. Please check the URL is valid.')
        content = [y.split(',') for y in req.content.decode('utf-8').split('\n') if len(y) > 1]
        for i in range(len(content)):
            for j in range(len(content[i])):
                if content[i][j] == '?':
                    continue
                content[i][j] = float(content[i][j])
        all_data.extend(content)

    # Can use pandas dataframes if we want
    df = pd.DataFrame(all_data, columns=columns)
    return all_data, df


def clean_df(df):
    # The following code is from the Jupyter Notebook by Avani and Lok
    df = df[df.fbs != '?']
    df=df[df.restecg!='?']
    df = df[df.chol != '?'].dropna()
    df=df[df.thalanch!='?'].dropna()
    df=df[df.exang!='?'].dropna()
    df=df[df.trestbps!='?'].dropna()
    df=df[df.oldpeak!='?'].dropna()

    df=df[df.trestbps>10]
    df=df[df.thalanch>60]
    df=df[df.chol!='?']
    df=df.drop(columns=['slope','ca','thal'])
    df=df[df.trestbps>50]
    df.thalanch=df.thalanch.astype('float64')

    # Converting to binary feature
    df.num = [int(x > 0) for x in df.num]

    return df


def get_data():
    _, df = pull_data_from_db()
    df = clean_df(df)
    features = [df[x].tolist() for x in df.columns[:-1]]
    features = [normalise_data(x) for x in features]
    feat = [[features[i][j] for i in range(len(features))] for j in range(len(features[0]))]
    labels = df['num'].tolist()
    feat_names = df.columns[:-1]
    return feat_names, feat, labels


def normalise_data(data):
    mind = min(data)
    maxd = max(data)
    norm = [(x - mind)/(maxd-mind) for x in data]
    return norm


def split_data(features, labels):
    # The .csv data comes pre-sorted so we need to shuffle the data around
    features, labels = shuffle(features, labels, random_state=69)

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
