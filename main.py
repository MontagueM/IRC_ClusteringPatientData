import data_handler
import model
import graph
import numpy as np
from sklearn.metrics import plot_roc_curve, precision_recall_curve, plot_precision_recall_curve
import matplotlib.pyplot as plt

params = {
    'axes.labelsize': 30,
    'font.size': 30,
    'legend.fontsize': 15,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'figure.figsize': [10, 8],
    'font.family': "Arial"
}

plt.rcParams.update(params)

if __name__ == '__main__':
    feature_names, features, labels = data_handler.get_data()

    training_features, training_labels, test_features, test_labels = data_handler.split_data(features, labels)

    svm_model = model.Model(training_features, training_labels, test_features, test_labels, feature_names)
    svm_model.train()
    svm_model.test()

    # confusion matrix and accuracy
    svm_model.evaluate()

    # ROC curve
    svc_disp = plot_roc_curve(svm_model.clf, svm_model.test_features, svm_model.test_labels)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
    plt.title("ROC curve plot")
    plt.savefig('figures/ROC.png')
    plt.show()

    svm_model.plot_bar_weights()

    # Five-fold cross validation scores
    scores,mean, se = data_handler.k_fold_cross_validation(k=5, kernel='linear', features=features, labels=labels)
    print(scores)
    print(mean)
    print(se)

    # 2D scatter plot
    a = 3
    b = 0
    features = [[x[a], x[b]] for x in features]
    feature_names = [feature_names[a], feature_names[b]]

    training_features, training_labels, test_features, test_labels = data_handler.split_data(features, labels)
    svm_model = model.Model(training_features, training_labels, test_features, test_labels, feature_names)
    svm_model.train()

    graph.visualise_clf(np.array(features), np.array(labels), svm_model.clf, feature_names)
