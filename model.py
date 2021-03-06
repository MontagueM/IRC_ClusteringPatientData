from sklearn import svm, metrics


def train_data(features, labels):
    clf = svm.SVC()
    clf.fit(features, labels)
    return clf


def test_data(clf, features):
    predictions = clf.predict(features)
    return predictions


def get_accuracy(true_labels, predicted_labels):
    return metrics.accuracy_score(true_labels, predicted_labels)