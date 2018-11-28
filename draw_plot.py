import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix

# import seaborn as sns

sns.set()
rcParams.update({'figure.autolayout': True})


def load_history(file_name):
    with open('hists/' + file_name, 'rb') as f:
        return pickle.load(f)


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


# %% 4B

hist_4b = load_history('4b')

# %% 4C
hists_4c = [hist_4b, load_history('4c-sigmoid'), load_history('4c-tanh')]
plt.clf()
plt.tight_layout()
ax = plt.gca()
for history in hists_4c:
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(history['acc'], color=color)
    plt.plot(history['val_acc'], '--', color=color)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

legends = []
for i in ['relu', 'sigmoid', 'tanh']:
    for j in ['train', 'test']:
        legends.append(' '.join((i, j)))

plt.legend(legends, loc='upper left', bbox_to_anchor=(0, -0.2),
           fancybox=True, shadow=True, ncol=3)
plt.title(" Comparison of activation functions")
# plt.show()
plt.savefig('plots/4c.png', dpi=300)

#%% 4D
hists_4d = [hist_4b, load_history('4d-sgd')]
plt.clf()
plt.tight_layout()
ax = plt.gca()
for history in hists_4d:
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(history['acc'], color=color)
    plt.plot(history['val_acc'], '--', color=color)

plt.ylabel('Accuracy')
plt.xlabel('Epoch')

legends = []
for i in ['adam', 'sgd']:
    for j in ['train', 'test']:
        legends.append(' '.join((i, j)))

plt.legend(legends, loc='upper left', bbox_to_anchor=(0, -0.2),
           fancybox=True, shadow=True, ncol=2)
plt.title("Comparison of Optimizer Algorithms")
plt.show()
# plt.savefig('plots/4d.png', dpi=300)
#%% 4E
hists_4e = [hist_4b, load_history('4e')]
plt.clf()
plt.tight_layout()
ax = plt.gca()
for history in hists_4e:
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(history['acc'], color=color)
    plt.plot(history['val_acc'], '--', color=color)

plt.ylabel('Accuracy')
plt.xlabel('Epoch')

legends = []
for i in ['60k samples', '600 samples']:
    for j in ['train', 'test']:
        legends.append(' '.join((i, j)))

plt.legend(legends, loc='upper left', bbox_to_anchor=(0, -0.2),
           fancybox=True, shadow=True, ncol=2)
plt.title("Comparison of Training size")
# plt.show()
plt.savefig('plots/4e.png', dpi=300)
#%% 4F
hists_4f = [hist_4b, load_history('4f-0.2'), load_history('4f-0.4'), load_history('4f-0.6')]
sns.set_palette("deep")
plt.clf()
plt.tight_layout()
ax = plt.gca()
for history in hists_4f:
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(history['acc'], color=color)
    plt.plot(history['val_acc'], '--', color=color)

plt.ylabel('Accuracy')
plt.xlabel('Epoch')

legends = []
for i in ['.0', '.2', '.4', '.6']:
    for j in ['train', 'test']:
        legends.append(' '.join((i, j)))

plt.legend(legends, loc='upper left', bbox_to_anchor=(0, -0.2),
           fancybox=True, shadow=True, ncol=4)
plt.title("Comparison of dropout probability")
# plt.show()
plt.savefig('plots/4f.png', dpi=300)

# %%
hists_4f2 = [hist_4b, load_history('4f2')]
plt.clf()
plt.tight_layout()
ax = plt.gca()
for history in hists_4f2:
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(history['acc'], color=color)
    plt.plot(history['val_acc'], '--', color=color)

plt.ylabel('Accuracy')
plt.xlabel('Epoch')

legends = []
for i in ['no data aug,', 'with data aug,']:
    for j in ['train', 'test']:
        legends.append(' '.join((i, j)))

plt.legend(legends, loc='upper left', bbox_to_anchor=(0, -0.2),
           fancybox=True, shadow=True, ncol=2)
plt.title("Effect of Data Augmentation on accuracy")
plt.show()
# plt.savefig('plots/4f2.png', dpi=300)