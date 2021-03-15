from sklearn import svm, metrics
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, auc


class Model:
    def __init__(self, train_features, train_labels, test_features, test_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels

        self.clf = None
        self.predicted_labels = None

    def train(self):
        self.clf = svm.SVC(kernel='linear')
        self.clf.fit(self.train_features, self.train_labels)
        return self.clf

    def test(self):
        self.predicted_labels = self.clf.predict(self.test_features)
        return self.predicted_labels

    def get_accuracy(self):
        return metrics.accuracy_score(self.test_labels, self.predicted_labels)

    def evaluate(self, b_confusion_matrix=True):
        if self.predicted_labels is None:
            self.test()
        accuracy = self.get_accuracy()
        print(f'Model accuracy: {round(accuracy * 100, 1)}%')
        print("**********************************************")
        if b_confusion_matrix:
            self.plot_confusion_matrix()

    def plot_confusion_matrix(self):
        print(metrics.confusion_matrix(self.test_labels, self.predicted_labels))
        sns.heatmap(metrics.confusion_matrix(self.test_labels, self.predicted_labels), annot=True, fmt=".0f", cmap="YlGnBu")
        plt.title("confusion matrix")
        plt.show()

    def plot_auc(self):
        train_fpr, train_tpr, tr_thresholds = roc_curve(self.test_labels, self.predicted_labels)
        test_fpr, test_tpr, te_thresholds = roc_curve(self.test_labels, self.predicted_labels)
        plt.grid()
        plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
        plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
        plt.plot([0,1],[0,1],'g--')
        plt.legend()
        plt.xlabel("True Positive Rate")
        plt.ylabel("False Positive Rate")
        plt.title("AUC(ROC curve)")
        plt.grid(color='black', linestyle='-', linewidth=0.5)
        plt.show()
