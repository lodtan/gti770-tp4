from benchmark.mlp import MLP
from sklearn.metrics import confusion_matrix
import numpy as np
from benchmark.benchmark import plot_confusion_matrix , get_scores
import matplotlib.pyplot as plt
import pandas as pd

path='./music'

benchmark_results =  pd.DataFrame(columns=['Model', 'Features','accuracy','f1_macro','precision_macro','recall_macro','precision_micro','recall_micro','f1_micro','type',
                                            'precision_weighted','recall_weighted','f1_weighted'])

for dataset in ['mfccs','trh','derivatives','mvd','moments','lpc','ssd','rh','spectral','marsyas']:

    mlp = MLP(dataset, path)
    #mlp.createWideModel()
    #mlp.createDeepModel()
    for type in ['deep','wide']:

        print('{0} MLP for {1}'.format(type,dataset))
        
        y_pred = mlp.predict_deep_model() if type == 'deep' else mlp.predict_wide_model()
    
        y_test = np.argmax(mlp.Y_test, axis=1)
        y_pred =  np.argmax(y_pred, axis=1)
        CM = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(CM, mlp.categories,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues)
        scores = get_scores(y_test,y_pred)
        scores['dataset'] = dataset
        scores['Model'] = 'MLP'
        scores['Features'] = mlp.X_train.shape[1]
        scores['type'] = type
        print('scores for {0} : {1} '.format(dataset,scores))
        benchmark_results = benchmark_results.append(scores,ignore_index=True)

benchmark_results.to_csv('{0}/Benchmark_Results_MLP_baseline.csv'.format('.'), index=False)