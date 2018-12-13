path='./music'
from benchmark.svm import SVM
from benchmark.processor import AudioProcessor
from benchmark.benchmark import plot_confusion_matrix , get_scores
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from decision.decision import Voting

dataset='lpc'
#dataset='mfcc'
#dataset='derivatives'

benchmark_results =  pd.DataFrame(columns=['Model', 'Features','accuracy','f1_macro','precision_macro','recall_macro','precision_micro','recall_micro','f1_micro',
                                           'precision_weighted','recall_weighted','f1_weighted'])

#for dataset in ['mfccs','trh','derivatives','mvd','moments','lpc','ssd','rh','spectral','marsyas']:
#for dataset in ['lpc','mfccs','derivatives']:

predictions = []
predictions_unlabeled = []

for dataset in ['lpc','mfcc','derivatives']:

    processor = AudioProcessor(dataset, path)
    X_train, X_test, Y_train, Y_test = processor.get_split_dataset(categorical=False)
    X_train, X_test = processor.scale_dataset(X_train, X_test)
    print('init SVM for {0} dataset'.format(dataset))
    categories = processor.inputs_le.inverse_transform(range(0, 25))
    print('Classes :  {0}'.format(categories))

    svm = SVM(X_train, X_test, Y_train, Y_test, dataset, './music')

    #svm.createSVMModel(X_train, Y_train, dataset)
    #clf2 = svm._load_svm_model(dataset)

    y_pred = svm.predict_svm_model(X_test,dataset)
    predictions.append(y_pred)

    CM = confusion_matrix(Y_test, y_pred)
    print(CM)
    plot_confusion_matrix(CM, categories,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues)
    scores = get_scores(Y_test,y_pred)
    scores['dataset'] = dataset
    scores['Model'] = 'SVM'
    scores['Features'] =  X_train.shape[1]
    print('scores for {0} : {1} '.format(dataset,scores))
    benchmark_results = benchmark_results.append(scores,ignore_index=True)

    # Unlabeled dataset
    df_untagged = processor._get_untagged_feature_vectors()
    features = np.asarray(df_untagged.values[:, 2:-1])
    track_ids = df_untagged.TRACKID.values

    scaler = StandardScaler()
    X_unlabeled = scaler.fit_transform(features)
    y_predict_unlabelled = svm.predict_svm_model(X_unlabeled,dataset)
    predictions_unlabeled.append(y_predict_unlabelled)

# write results in csv file
benchmark_results.to_csv('{0}/Benchmark_Results_SVM_baseline.csv'.format('.'), index=False)
# read results from csv file

# Saved Labeled predictions
final = np.concatenate(predictions,axis=1)
df_final = pd.DataFrame(final)
df_final.to_csv('{0}/SVM_Predictions_test.csv'.format('.'), index=False)

# Save Unlabeled predictions
final_unlabeled = np.concatenate(predictions_unlabeled,axis=1)
df_final_unlabeled = pd.DataFrame(final_unlabeled)
df_final_unlabeled['TRACKID'] = pd.Series(track_ids)
df_final_unlabeled.to_csv('{0}/SVM_Predictions_unlabeled.csv'.format('.'), index=False)


categories = ('BIG_BAND', 'BLUES_CONTEMPORARY', 'COUNTRY_TRADITIONAL', 'DANCE', 'ELECTRONICA', 'EXPERIMENTAL',
              'FOLK_INTERNATIONAL', 'GOSPEL', 'GRUNGE_EMO', 'HIP_HOP_RAP', 'JAZZ_CLASSIC', 'METAL_ALTERNATIVE',
              'METAL_DEATH', 'METAL_HEAVY', 'POP_CONTEMPORARY', 'POP_INDIE', 'POP_LATIN', 'PUNK', 'REGGAE',
              'RNB_SOUL', 'ROCK_ALTERNATIVE', 'ROCK_COLLEGE', 'ROCK_CONTEMPORARY', 'ROCK_HARD', 'ROCK_NEO_PSYCHEDELIA')

STRATEGIES = ('sum', 'max', 'min')

predictions_unlabeled = pd.read_csv('{0}/SVM_Predictions_unlabeled.csv'.format('.'), sep=',')

voter = Voting('.')

df = voter.SVMdecision(categories, STRATEGIES[0])
print(df.shape)


