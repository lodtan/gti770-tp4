from benchmark.mlp import MLP
from sklearn.metrics import confusion_matrix
import numpy as np
from benchmark.benchmark import plot_confusion_matrix
import matplotlib.pyplot as plt

path='./music'
#for dataset in ['mfccs','trh','derivatives','mvd','moments','lpc','ssd','rh','spectral','marsyas']:
for dataset in ['marsyas']:

    mlp = MLP(dataset, path)
    #mlp.createWideModel()
    #mlp.createDeepModel()
    y_pred = mlp.predict_deep_model()

    CM = confusion_matrix(np.argmax(mlp.Y_test, axis=1), np.argmax(y_pred, axis=1))
    plot_confusion_matrix(CM, mlp.categories,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues)

#mlp=MLP('wide',dataset,path)
#mlp._createWideModel()