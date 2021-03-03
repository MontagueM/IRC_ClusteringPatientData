import pandas as pd

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
        'sex',
        'trestbps',
        'target'
    ]
    df = df[df.columns.intersection(desired_cols)]
    # Forming a feature/label array for ease of use with svm. Code is not nice but basically just converts to two arrays of the relevant features and labels needed for SVM classification
    features = [df[x].tolist() for x in desired_cols[:-1]]
    feat = [[features[i][j] for i in range(len(features))] for j in range(len(features[0]))]
    labels = df[desired_cols[-1]].tolist()
    return feat, labels


if __name__ == '__main__':
    features, labels = get_data()
