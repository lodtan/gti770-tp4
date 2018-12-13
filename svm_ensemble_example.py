path='./music'
from benchmark.svm import SVM
from benchmark.processor import AudioProcessor
from benchmark.benchmark import plot_confusion_matrix , get_scores
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

dataset='lpc'
#dataset='mfcc'
#dataset='derivatives'

benchmark_results =  pd.DataFrame(columns=['Model', 'Features','accuracy','f1_macro','precision_macro','recall_macro','precision_micro','recall_micro','f1_micro',
                                           'precision_weighted','recall_weighted','f1_weighted'])

#for dataset in ['mfccs','trh','derivatives','mvd','moments','lpc','ssd','rh','spectral','marsyas']:
for dataset in ['lpc','mfccs','derivatives']:

    processor = AudioProcessor(dataset, path)
    X_train, X_test, Y_train, Y_test = processor.get_split_dataset(categorical=False)
    X_train, X_test = processor.scale_dataset(X_train, X_test)
    print('init SVM for {0} dataset'.format(dataset))
    categories = processor.inputs_le.inverse_transform(range(0, 25))
    print('Classes :  {0}'.format(categories))

    svm = SVM(X_train, X_test, Y_train, Y_test, dataset, './music')

    svm.createSVMModel(X_train, Y_train, dataset)
    clf2 = svm._load_svm_model('lpc')

    y_pred = svm.predict_svm_model(X_test,dataset)

    CM = confusion_matrix(Y_test, y_pred)
    print(CM)
    plot_confusion_matrix(CM, categories,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues)
    scores = get_scores(Y_test,y_pred)
    scores['dataset'] = dataset
    scores['Model'] = 'SVM'
    scores['Features'] =  X_train.shape[1]
    print('scores for {0} : {1} '.format(dataset,scores))
    benchmark_results = benchmark_results.append(scores,ignore_index=True)

# write results in csv file
benchmark_results.to_csv('{0}/Benchmark_Results_SVM_baseline.csv'.format('.'), index=False)
# read results from csv file


