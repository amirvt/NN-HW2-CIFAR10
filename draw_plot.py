import itertools
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
# import seaborn as sns

from matplotlib import rcParams
import seaborn as sns

sns.set()
rcParams.update({'figure.autolayout': True})


def plot_confusion_matrix(model, x_test, y_test, title, classes):
    plt.clf()
    plt.tight_layout()
    predictions = model.predict(x_test)

    print('Confusion matrix (rows: true classes; columns: predicted classes):')
    print()
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1), labels=classes)
    print(cm)
    print()

    print('Classification accuracy for each class:')
    print()
    for i, j in enumerate(cm.diagonal() / cm.sum(axis=1)): print("%d: %.4f" % (i, j))

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print_confusion_matrix(cm, classes)

    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title(title)
    # plt.colorbar()
    # tick_marks = np.arange(10)
    # plt.xticks(tick_marks, range(10))
    # plt.yticks(tick_marks, range(10))
    #
    # fmt = '.2f'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")
    #
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.tight_layout()

    # plt.show()
    plt.title('Confusion matrix for ' + title)
    plt.savefig('plots/' + title + ' conf matrix' + '.png')


def print_confusion_matrix(confusion_matrix, class_names, figsize=(8, 6), fontsize=14, normalize=True):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f" if normalize else 'd', cmap=sns.cm.rocket_r)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def plot_history(histories, titles):
    for history, title in zip(histories, titles):
        plt.clf()
        plt.tight_layout()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.title(title + " Accuracy")
        # plt.show()
        plt.savefig('plots/' + title + 'accuracy .png')
