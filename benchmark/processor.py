import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize, StandardScaler

class AudioProcessor(object):

    """ Return Audio Feature Vectors and labels."""

    def __init__(self, dataset, path='../music'):
        self._path = path
        self.dataset = dataset
        self._exts = '.csv'
        self.inputs_le = LabelEncoder()
        self.scaler = StandardScaler()

        filenames = ['msd-jmirmfccs_dev','msd-trh_dev','msd-jmirderivatives_dev','msd-mvd_dev','msd-jmirmoments_dev',
                     'msd-jmirlpc_dev','msd-ssd_dev','msd-rh_dev_new','msd-jmirspectral_dev','msd-marsyas_dev_new']

        datasets = ['mfccs','trh','derivatives','mvd','moments','lpc','ssd','rh','spectral','marsyas']

        file_mappings = dict(zip(datasets,filenames))

        if dataset not in datasets:
            raise ValueError("dataset parameter should be in {0}".format(datasets))
        else:
            self.dataset = dataset
            self.filename = file_mappings[dataset]

    def _get_labels(self):
        print('[INFO] Reading attributes {0}'.format(self.dataset))
        df = pd.read_csv('{0}/tagged_feature_sets/{1}/attributes.txt'.format(self._path, self.filename), sep=' ', delimiter=None, encoding='utf-8',names=[0, 1])
        return df

    def _get_feature_vectors(self):
        print('[INFO] Reading dataset {0}'.format(self.dataset))

        df = pd.read_csv('{0}/tagged_feature_sets/{1}/{1}.csv'.format(self._path, self.filename), sep=',', delimiter=None, index_col=None,
                         encoding='utf-8', names=self._get_labels()[0].values)

        return df

    def get_split_dataset(self):

        df = self._get_feature_vectors()

        labels = self.inputs_le.fit_transform(df['class'].values)
        labels = to_categorical(labels, num_classes=df['class'].nunique())

        features = np.asarray(df.values[:, 2:-1])

        print("[INFO] Labels format : {0}".format(labels.shape))
        print("[INFO] Features format : {0}".format(features.shape))

        print('[INFO] Splitting dataset {0}'.format(self.dataset))
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels,test_size=0.20, random_state=42,stratify=labels)
        print("[INFO] Labels format : {0}".format(labels.shape))
        print("[INFO] Features format : {0}".format(X_train.shape))
        return X_train, X_test, Y_train, Y_test

    def scale_dataset(self,X, X_test):
        print("[INFO] Scaler : removing the mean and scaling to unit variance")
        X = self.scaler.fit_transform(X)
        X_test = self.scaler.fit_transform(X_test)
        return X, X_test




