import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


CATEGORIES = ('BIG_BAND', 'BLUES_CONTEMPORARY', 'COUNTRY_TRADITIONAL', 'DANCE', 'ELECTRONICA', 'EXPERIMENTAL',
              'FOLK_INTERNATIONAL', 'GOSPEL', 'GRUNGE_EMO', 'HIP_HOP_RAP', 'JAZZ_CLASSIC', 'METAL_ALTERNATIVE',
              'METAL_DEATH', 'METAL_HEAVY', 'POP_CONTEMPORARY', 'POP_INDIE', 'POP_LATIN', 'PUNK', 'REGGAE',
              'RNB_SOUL', 'ROCK_ALTERNATIVE', 'ROCK_COLLEGE', 'ROCK_CONTEMPORARY', 'ROCK_HARD', 'ROCK_NEO_PSYCHEDELIA')

DATASETS = ('derivatives', 'lpc', 'mfccs', 'moments', 'spectral', 'marsyas', 'mvd', 'rh', 'ssd', 'trh')


def loadDataset(dataset):
    assert dataset in DATASETS
    print("[INFO] loading {0} dataset ...".format(dataset))
    if dataset in ('derivatives', 'lpc', 'mfccs', 'moments', 'spectral'):
        dataset_name = 'jmir'+dataset
    else:
        dataset_name = dataset
    if dataset in ('marsyas', 'rh'):
        new = '_new'
    else:
        new = ''
    IDs = pd.read_csv('tagged_feature_sets/msd-'+dataset_name+'_dev'+new+'/msd-'+dataset_name+'_dev'+new+'.csv',
                      delimiter=',', header=None).values[:, 1]
    labels = pd.read_csv('tagged_feature_sets/msd-'+dataset_name+'_dev'+new+'/msd-'+dataset_name+'_dev'+new+'.csv',
                         delimiter=',', header=None).values[:, -1:]
    for i, label in enumerate(labels):
        labels[i] = CATEGORIES.index(label)
    features = pd.read_csv('tagged_feature_sets/msd-'+dataset_name+'_dev'+new+'/msd-'+dataset_name+'_dev'+new+'.csv',
                           delimiter=',', header=None).values[:, 2:-1]
    np.asarray(labels)
    np.asarray(features)
    labels = to_categorical(labels, num_classes=25)
    print("[INFO] splitting the data ...")
    X, X_test, Y, Y_test = train_test_split(features, labels, test_size=0.20, random_state=42, stratify=labels)
    print("[INFO] Labels format : {0}".format(labels.shape))
    print("[INFO] Features format : {0}".format(features.shape))
    return X, X_test, Y, Y_test, IDs


def scaleDataset(*targets):
    print("[INFO] Scaling {0} sets...".format(len(targets)))
    scaler = StandardScaler()
    scaledDatasets = ()
    for set in targets:
        scaledDatasets = scaledDatasets + (scaler.fit_transform(set),)
    return scaledDatasets


def getNumFeatures(data):
    shape = data.shape
    if len(shape) == 1:
        shape = shape[0]
    else:
        shape = shape[-1]
    print("[INFO] This data have {0} features.".format(shape))
    return shape


if __name__ == "__main__":
    X, X_test, Y, Y_test, IDs = loadDataset(DATASETS[0])
    X, X_test = scaleDataset(X, X_test)
    getNumFeatures(X[0])
    getNumFeatures(X)
