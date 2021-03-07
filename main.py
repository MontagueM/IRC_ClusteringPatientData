import data_handler
import model
import graph
import numpy as np

if __name__ == '__main__':
    feature_names, features, labels = data_handler.get_data()

    # Uncomment if visualising data
    #features = [x[:2] for x in features]
    #feature_names = feature_names[:2]

    training_features, training_labels, test_features, test_labels = data_handler.split_data(features, labels)

    svm_clf = model.train_data(training_features, training_labels)
    predictions = model.test_data(svm_clf, test_features)
    accuracy = model.get_accuracy(test_labels, predictions)

    #uncomment to get 5 cross validation scores
    #scores = data_handler.k_fold_cross_validation(k=5, kernel= 'linear',features=features,labels=labels)
    #print(scores)
    
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #(I'm not completely sure if the features and labels are used correctly in this implementation but
    #the output seems ok. Still am unsure of the interpretation of the scores.)

    print(f'Model accuracy: {round(accuracy*100, 1)}%')

    #graph.visualise_clf(np.array(features), np.array(labels), svm_clf, feature_names)
