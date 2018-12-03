import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(2)

    plt.subplots(figsize=(20, 15))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, cmap=cmap, annot=True,
                xticklabels=classes,
                yticklabels=classes)

    plt.title(title)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')

    plt.show()
    plt.close()