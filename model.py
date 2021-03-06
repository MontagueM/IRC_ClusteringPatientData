from sklearn import svm, metrics


def train_data(features, labels):
    clf = svm.SVC(kernel='linear')
    clf.fit(features, labels)
    return clf


def test_data(clf, features):
    predictions = clf.predict(features)
    return predictions


def get_accuracy(true_labels, predicted_labels):
    return metrics.accuracy_score(true_labels, predicted_labels)


"""
Things to look into:
- ROC curve (prediction vs random chance)
- p-hacking (same data for multiple hypothesis)
"""