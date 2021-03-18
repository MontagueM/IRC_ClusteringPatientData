import data_handler
import model
import graph
import numpy as np
from sklearn.metrics import plot_roc_curve, precision_recall_curve, plot_precision_recall_curve
import matplotlib.pyplot as plt

if __name__ == '__main__':
    feature_names, features, labels = data_handler.get_data()

    training_features, training_labels, test_features, test_labels = data_handler.split_data(features, labels)

    svm_model = model.Model(training_features, training_labels, test_features, test_labels)
    svm_model.train()
    svm_model.test()

    # confusion matrix and accuracy
    svm_model.evaluate()

    # ROC curve
    svc_disp = plot_roc_curve(svm_model.clf, svm_model.test_features, svm_model.test_labels)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
    plt.title("ROC curve plot")
    plt.show()
   
    # Five-fold cross validation scores
    scores = data_handler.k_fold_cross_validation(k=5, kernel='linear', features=features, labels=labels)
    print(scores)
    


    # 2D scatter plot
    features = [x[:2] for x in features]
    feature_names = feature_names[:2]

    training_features, training_labels, test_features, test_labels = data_handler.split_data(features, labels)
    svm_model = model.Model(training_features, training_labels, test_features, test_labels)
    svm_model.train()

    graph.visualise_clf(np.array(features), np.array(labels), svm_model.clf, feature_names)
