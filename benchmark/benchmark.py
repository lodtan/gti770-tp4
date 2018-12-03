import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score , f1_score , recall_score


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues,display=False):
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
        cm = cm.round(3)

    plt.subplots(figsize=(20, 15))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, cmap=cmap, annot=True,
                xticklabels=classes,
                yticklabels=classes)

    plt.title(title)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')

    if display:
        plt.show()
    #plt.close()

def get_scores(y_test,y_test_pred):

    accuracy = accuracy_score(y_test, y_test_pred)
    recall_macro = recall_score(y_test, y_test_pred, average='macro')
    precision_macro = precision_score(y_test, y_test_pred, average='macro')
    f1_macro = f1_score(y_test, y_test_pred, average='macro')

    recall_micro = recall_score(y_test, y_test_pred, average='micro')
    precision_micro = precision_score(y_test, y_test_pred, average='micro')
    f1_micro = f1_score(y_test, y_test_pred, average='micro')

    recall_weighted = recall_score(y_test, y_test_pred, average='weighted')
    precision_weighted = precision_score(y_test, y_test_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

    return {'accuracy': accuracy,'f1_macro': f1_macro,'precision_macro': precision_macro,
            'recall_macro': recall_macro,'f1_micro': f1_micro,'precision_micro': precision_micro,'recall_micro': recall_micro,
            'recall_weighted': recall_weighted,'f1_weighted': f1_weighted,'precision_weighted': precision_weighted}