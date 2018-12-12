import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.svm import SVC

class SVM(object):

    def __init__(self, X_train, X_test, Y_train, Y_test, dataset, path, epochs=50, batch_size=500, init_lr=5e-4):

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        self.num_classes = 25
        self.num_features = self.X_train.shape[1]

        self.dataset = dataset
        self.path = path

        if os.path.exists('{0}/ModelsSVM/SVM{1}.hdf5'.format(self.path, self.dataset)):
            print('[INFO] loading svm model')
            self.deep_model = self.load_svm_model()

    def createSVMModel(self):

        clf.save('{0}/ModelsMLP/deepMLP{1}.hdf5'.format(self.path, self.dataset))
        return

    def _load_model(self, type):
        return load_model('{0}/ModelsSVM/{1}MLP{2}.hdf5'.format(self.path, type, self.dataset))

    def load_svm_model(self):
        return self._load_model('deep')

    def predict_svm_model(self):
        y_predict = self.deep_model.predict(self.X_test)
        return y_predict

    def _grid_search(self,model,param_grid,default_params,X_train, y_train):

        model_name = model.__name__
        self.logger.info('{0} GRID search for : {1} '.format(time.strftime("%Y%m%d:%H:%M:%S"), model_name))

        # add default paramaters for each model
        grid_search = GridSearchCV(estimator=model(), param_grid=param_grid, cv=self.tscv.split(X_train), n_jobs=6,verbose=4,scoring='f1_macro')
        grid_search.fit(X_train, y_train)

        self.logger.info('{0} Best parameters set found on training set: '.format(time.strftime("%Y%m%d:%H:%M:%S")))
        self.logger.info('{0} {1}'.format(time.strftime("%Y%m%d:%H:%M:%S"),grid_search.best_params_))
        self.logger.info('Grid scores on training set:'.format(time.strftime("%Y%m%d:%H:%M:%S"), grid_search.best_params_))

        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']

        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            self.logger.info("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

        #res=pd.DataFrame([[model_name,grid_search.best_params_]],columns=['model','best_params'])
        res= grid_search.best_params_

        #self.best_params[model_name] = grid_search.best_params_

        return res