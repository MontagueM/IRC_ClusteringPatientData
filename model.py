from sklearn import svm, metrics
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, auc


def train_data(features, labels):
    clf = svm.SVC(kernel='linear')
    clf.fit(features, labels)
    return clf


def test_data(clf, features):
    predictions = clf.predict(features)
    return predictions


def get_accuracy(true_labels, predicted_labels):
    return metrics.accuracy_score(true_labels, predicted_labels)

def evaluation(true_labels, predicted_labels):
    accuracy = get_accuracy(true_labels,predicted_labels)
    print(f'Model accuracy: {round(accuracy*100, 1)}%')
    print("**********************************************")
    print(metrics.confusion_matrix(true_labels, predicted_labels))
    sns.heatmap(metrics.confusion_matrix(true_labels, predicted_labels), annot = True, fmt = ".0f", cmap = "YlGnBu")
    plt.title("confusion matrix")
    plt.show()
    
# def auc_plot(test_labels,predictions):
#     train_fpr, train_tpr, tr_thresholds = roc_curve(test_labels, predictions)
#     test_fpr, test_tpr, te_thresholds = roc_curve(test_labels, predictions)
#     plt.grid()
#     plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
#     plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
#     plt.plot([0,1],[0,1],'g--')
#     plt.legend()
#     plt.xlabel("True Positive Rate")
#     plt.ylabel("False Positive Rate")
#     plt.title("AUC(ROC curve)")
#     plt.grid(color='black', linestyle='-', linewidth=0.5)
#     plt.show()


"""
Things to look into:
- ROC curve (prediction vs random chance)
- p-hacking (same data for multiple hypothesis)
"""