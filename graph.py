from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# https://stackoverflow.com/questions/51297423/plot-scikit-learn-sklearn-svm-decision-boundary-surface


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def stack_overflow(features, labels, clf, feature_names):
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of linear SVC')
    # Set-up grid for plotting.
    X0, X1 = features[:, 0], features[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel(feature_names[0])
    ax.set_xlabel(feature_names[1])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.savefig('figures/decision_surface.png',transparent=True)
    plt.show()

# End of stackoverflow code


def visualise_clf(features, labels, clf, feature_names):
    # Max two features
    if len(features[0]) != 2:
        raise Exception('Only two features must be given to visualise a 2D plot.')

    stack_overflow(features, labels, clf, feature_names)
