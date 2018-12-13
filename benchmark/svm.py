import numpy as np
from sklearn.decomposition import PCA
import os
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score , GridSearchCV , StratifiedShuffleSplit
import json
from functools import wraps
from joblib import dump, load

def check_cache(tag):

    def cache_decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):

            _self = args[0]
            ds = args[3]
            filename=tag+'_'+ds

            if _self._is_file_cached(filename) :
                print('[INFO] reading {0} from cache'.format(filename))
                res = _self._get_cached_file(filename)
            else:
                print('[INFO] No Cache found for {0} => calling the function {1}'.format(tag,func.__name__))
                res = func(*args)

            return res

        return func_wrapper

    return cache_decorator


class SVM(object):

    def __init__(self, X_train, X_test, Y_train, Y_test, dataset, path):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        self.num_classes = 25
        self.num_features = self.X_train.shape[1]

        self.dataset = dataset
        self.path = path

        self.best_params = {'lpc': self.svm_grid_search(X_train,Y_train,'lpc'),
                            'mfcc': self.svm_grid_search(X_train, Y_train, 'mfcc'),
                            'derivatives': self.svm_grid_search(X_train, Y_train, 'derivatives'),
                            'trh': self.svm_grid_search(X_train, Y_train, 'trh'),
                            'mvd': self.svm_grid_search(X_train, Y_train, 'mvd'),
                            'moments': self.svm_grid_search(X_train, Y_train, 'moments'),
                            'ssd': self.svm_grid_search(X_train, Y_train, 'ssd'),
                            'rh': self.svm_grid_search(X_train, Y_train, 'rh'),
                            'spectral': self.svm_grid_search(X_train, Y_train, 'spectral'),
                            'marsyas': self.svm_grid_search(X_train, Y_train, 'marsyas')}

        print('[INFO] best parameters for SVM : {0} '.format(self.best_params))

    def createSVMModel(self,X_train,Y_train,dataset):

        X_train = self._apply_pca(X_train)

        model_params = {**self.best_params[dataset], **{'cache_size': 2048}}
        clf = SVC(**model_params)
        print('[INFO] Training SVM Model for {0} dataset'.format(dataset))
        clf.fit(X_train,Y_train)
        print('[INFO] Saving SVM Model for {0} dataset'.format(dataset))
        dump(clf, '{0}/ModelsSVM/SVM_{1}.joblib'.format(self.path,dataset))
        return clf

    def _load_svm_model(self,dataset):
        print('[INFO] Loading SVM Model for {0} dataset'.format(dataset))
        clf = load('{0}/ModelsSVM/SVM_{1}.joblib'.format(self.path,dataset))
        return clf

    def predict_svm_model(self,X_test,dataset):
        X_test = self._apply_pca(X_test)
        clf = self._load_svm_model(dataset)
        y_predict = clf.predict(X_test)
        return y_predict

    def _grid_search(self,model,model_params,param_grid,X_train, y_train,k=2):

        model_name = model.__name__
        print('{0} GRID search for : {1} '.format(time.strftime("%Y%m%d:%H:%M:%S"), model_name))

        # add default paramaters for each model
        grid_search = GridSearchCV(estimator=model(**model_params), param_grid=param_grid,cv=k, n_jobs=8,verbose=5,scoring='f1_macro')
        grid_search.fit(X_train, y_train)

        print('{0} Best parameters set found on training set: '.format(time.strftime("%Y%m%d:%H:%M:%S")))
        print('{0} {1}'.format(time.strftime("%Y%m%d:%H:%M:%S"),grid_search.best_params_))
        print('Grid scores on training set:'.format(time.strftime("%Y%m%d:%H:%M:%S"), grid_search.best_params_))

        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']

        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

        return  grid_search.best_params_

    @check_cache('SVM')
    def svm_grid_search(self,X_train,y_train,dataset):

        X_train = self._apply_pca(X_train)

        model_params = {'cache_size':2048}
        param_grid_rbf = {'kernel': ['rbf'], 'C': [10 ** (-3),1,10 ** (-1), 10], 'gamma': [10 ** (-3), 10 ** (-1),1, 10]}
        res = self._grid_search(SVC,model_params,param_grid_rbf,X_train,y_train,k=2)

        self._save_cache(res, 'SVM_{0}'.format(dataset))

    def _apply_pca(self, X):

        print('[INFO] Applying PCA on dataset of {0} dimensions'.format(X.shape[1]))
        pca = PCA(0.9, svd_solver='full')
        principalComponents = pca.fit_transform(X)
        print('[INFO] PCA completed feature space reduced to {0} dimensions'.format(principalComponents.shape[1]))
        return principalComponents

    #cache
    def _save_cache(self,df,filename):

        with open('{0}/ModelsSVM/{1}.json'.format(self.path,filename), 'w') as outfile:
            json.dump(df, outfile)

    def _is_file_cached(self,filename):
        return os.path.isfile('{0}/ModelsSVM/{1}.json'.format(self.path,filename))

    def _get_cached_file(self,filename):

        with open('{0}/ModelsSVM/{1}.json'.format(self.path, filename), 'r') as infile:
            return json.load(infile)

