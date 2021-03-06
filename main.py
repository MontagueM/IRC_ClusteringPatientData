import data_handler
import model


if __name__ == '__main__':
    features, labels = data_handler.get_data()
    training_features, training_labels, test_features, test_labels = data_handler.split_data(features, labels)

    svm_clf = model.train_data(training_features, training_labels)
    predictions = model.test_data(svm_clf, test_features)
    accuracy = model.get_accuracy(test_labels, predictions)
    print(f'Model accuracy: {round(accuracy*100, 1)}%')
