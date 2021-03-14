import data_handler
import model
import graph
import numpy as np
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt

if __name__ == '__main__':
    feature_names, features, labels = data_handler.get_data()

    #Uncomment if visualising data
    #features = [x[:2] for x in features]
    #feature_names = feature_names[:2]

    training_features, training_labels, test_features, test_labels = data_handler.split_data(features, labels)

    svm_clf = model.train_data(training_features, training_labels)
    predictions = model.test_data(svm_clf, test_features)
    accuracy = model.get_accuracy(test_labels, predictions)

    print(f'Model accuracy: {round(accuracy*100, 1)}%')

    #uncomment for confusion matrix and accuracy 
    #model.evaluation(test_labels, predictions)


    #uncomment for ROC curve and AUC score
    #svc_disp = plot_roc_curve(svm_clf, test_features, test_labels)
    #plt.show()
   
    #uncomment to get 5 cross validation scores
    #scores = data_handler.k_fold_cross_validation(k=5, kernel= 'linear',features=features,labels=labels)
    #print(scores)


    #model.auc_plot(test_features, test_labels)

    #uncomment for 2D scatter plot
    #graph.visualise_clf(np.array(features), np.array(labels), svm_clf, feature_names)
