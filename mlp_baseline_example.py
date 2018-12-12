from benchmark.mlp import MLP
from sklearn.metrics import confusion_matrix
import numpy as np
from benchmark.benchmark import plot_confusion_matrix , get_scores
import matplotlib.pyplot as plt
import pandas as pd
from benchmark.processor import AudioProcessor
from sklearn.preprocessing import StandardScaler
path='./music'
from decision.decision import Voting

benchmark_results =  pd.DataFrame(columns=['Model', 'Features','accuracy','f1_macro','precision_macro','recall_macro','precision_micro','recall_micro','f1_micro','type',
                                           'precision_weighted','recall_weighted','f1_weighted'])

# for dataset in ['mfccs','trh','derivatives','mvd','moments','lpc','ssd','rh','spectral','marsyas']:
#
#     processor = AudioProcessor(dataset, path)
#     X_train, X_test, Y_train, Y_test = processor.get_split_dataset()
#     X_train, X_test = processor.scale_dataset(X_train, X_test)
#     print('init MLPs for {0} dataset'.format(dataset))
#     categories = processor.inputs_le.inverse_transform(range(0, 25))
#     print('Classes :  {0}'.format(categories))
#
#     mlp = MLP(X_train, X_test, Y_train, Y_test,dataset, path)
#     #mlp.createWideModel()
#     #mlp.createDeepModel()
#     for type in ['deep','wide']:
#
#         print('{0} MLP for {1}'.format(type,dataset))
#
#         y_pred = mlp.predict_deep_model() if type == 'deep' else mlp.predict_wide_model()
#
#         y_test = np.argmax(mlp.Y_test, axis=1)
#         y_pred =  np.argmax(y_pred, axis=1)
#         CM = confusion_matrix(y_test, y_pred)
#         print(CM)
#         categories = processor.inputs_le.inverse_transform(range(0, 25))
#         plot_confusion_matrix(CM, categories,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues)
#         scores = get_scores(y_test,y_pred)
#         scores['dataset'] = dataset
#         scores['Model'] = 'MLP'
#         scores['Features'] = mlp.X_train.shape[1]
#         scores['type'] = type
#         print('scores for {0} : {1} '.format(dataset,scores))
#         benchmark_results = benchmark_results.append(scores,ignore_index=True)

# write results in csv file
#benchmark_results.to_csv('{0}/Benchmark_Results_MLP_baseline.csv'.format('.'), index=False)
# read results from csv file


def get_best_n_dataset(n=3):

    benchmark_results = pd.read_csv('{0}/Benchmark_Results_MLP_baseline.csv'.format('.'), sep=',')
    bestn = benchmark_results.sort_values(['accuracy'],ascending=False)[['dataset','type']].drop_duplicates(subset='dataset',keep='first').head(n)
    selected_datasets = bestn.dataset.values
    selected_mlp_type = bestn.type.values
    print('Selected datasets {0} Selected {1}'.format(selected_datasets,selected_mlp_type))
    return selected_datasets,selected_mlp_type

N=3
selected_datasets , selected_mlp_type = get_best_n_dataset(n=N)

predictions = []
predictions_unlabeled = []

#predict for unlabeled dataset and test
for mlp_model in range(0,N):

    myprocessor = AudioProcessor(selected_datasets[mlp_model], path)
    X_train, X_test, Y_train, Y_test = myprocessor.get_split_dataset()
    X_train, X_test = myprocessor.scale_dataset(X_train, X_test)
    print('init MLPs for {0} dataset'.format(selected_datasets[mlp_model]))
    categories = myprocessor.inputs_le.inverse_transform(range(0, 25))
    print('Classes :  {0}'.format(categories))

    mlp = MLP(X_train, X_test, Y_train, Y_test,selected_datasets[mlp_model], path)
    print(selected_datasets[mlp_model],selected_mlp_type[mlp_model])

    # Labeled data
    y_pred = mlp.predict_deep_model() if selected_mlp_type[mlp_model] == 'deep' else mlp.predict_wide_model()
    print(y_pred.shape)
    predictions.append(y_pred)

    # Unlabeled dataset
    df_untagged = myprocessor._get_untagged_feature_vectors()
    features = np.asarray(df_untagged.values[:, 2:-1])
    track_ids = df_untagged.TRACKID.values

    scaler = StandardScaler()
    X_unlabeled = scaler.fit_transform(features)
    y_predict_unlabelled = mlp.wide_model.predict(X_unlabeled)
    predictions_unlabeled.append(y_predict_unlabelled)

# Saved Labeled predictions
final = np.concatenate(predictions,axis=1)
df_final = pd.DataFrame(final)
df_final.to_csv('{0}/MLP_Predictions_test.csv'.format('.'), index=False)

# Save Unlabeled predictions
final_unlabeled = np.concatenate(predictions_unlabeled,axis=1)
df_final_unlabeled = pd.DataFrame(final_unlabeled)
df_final_unlabeled['TRACKID'] = pd.Series(track_ids)
df_final_unlabeled.to_csv('{0}/MLP_Predictions_unlabeled.csv'.format('.'), index=False)


categories = ('BIG_BAND', 'BLUES_CONTEMPORARY', 'COUNTRY_TRADITIONAL', 'DANCE', 'ELECTRONICA', 'EXPERIMENTAL',
              'FOLK_INTERNATIONAL', 'GOSPEL', 'GRUNGE_EMO', 'HIP_HOP_RAP', 'JAZZ_CLASSIC', 'METAL_ALTERNATIVE',
              'METAL_DEATH', 'METAL_HEAVY', 'POP_CONTEMPORARY', 'POP_INDIE', 'POP_LATIN', 'PUNK', 'REGGAE',
              'RNB_SOUL', 'ROCK_ALTERNATIVE', 'ROCK_COLLEGE', 'ROCK_CONTEMPORARY', 'ROCK_HARD', 'ROCK_NEO_PSYCHEDELIA')

STRATEGIES = ('sum', 'max', 'min')

predictions_unlabeled = pd.read_csv('{0}/MLP_Predictions_unlabeled.csv'.format('.'), sep=',')

voter = Voting('.')

df = voter.MLPdecision(categories, STRATEGIES[0])
print(df.shape)