import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.layers import BatchNormalization, Dropout
import os


class SVM(object):

    def __init__(self, X_train, X_test, Y_train, Y_test, dataset, path, epochs=50, batch_size=500, init_lr=5e-4):

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        self.num_classes = 25
        self.num_features = self.X_train.shape[1]
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_lr = init_lr

        self.dataset = dataset
        self.path = path

        if os.path.exists('{0}/ModelsMLP/deepMLP{1}.hdf5'.format(self.path, self.dataset)):
            print('[INFO] loading deep model')
            self.deep_model = self.load_deep_model()

        if os.path.exists('{0}/ModelsMLP/wideMLP{1}.hdf5'.format(self.path, self.dataset)):
            print('[INFO] loading wide model')
            self.wide_model = self.load_wide_model()

    def createWideModel(self):

        plots = True
        wideModel = self._wideMLP(self.num_classes)
        wideModel.compile(optimizer=Adam(lr=self.init_lr, decay=self.init_lr / self.epochs),
                          loss='categorical_crossentropy', metrics=['accuracy'])
        print("[INFO] training wide model on {0} dataset".format(self.dataset))
        wideH = wideModel.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size,
                              validation_data=(self.X_test, self.Y_test), verbose=0)
        wideModel.save('{0}/ModelsMLP/wideMLP{1}.hdf5'.format(self.path, self.dataset))

        if plots:
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(np.arange(0, self.epochs), wideH.history["loss"], label="train_loss")
            plt.plot(np.arange(0, self.epochs), wideH.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, self.epochs), wideH.history["acc"], label="train_acc")
            plt.plot(np.arange(0, self.epochs), wideH.history["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy on " + self.dataset)
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            plt.savefig('{0}/GraphsMLP/wideMLP{1}.png'.format(self.path, self.dataset))
        self.wide_model = wideModel
        return wideH.history["val_acc"][-1]

    def createDeepModel(self):

        plots = True
        deepModel = self._deepMLP(self.num_classes)
        deepModel.compile(optimizer=Adam(lr=self.init_lr, decay=self.init_lr / self.epochs),
                          loss='categorical_crossentropy', metrics=['accuracy'])
        print("[INFO] training deep model on {0} dataset".format(self.dataset))
        deepH = deepModel.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size,
                              validation_data=(self.X_test, self.Y_test), verbose=0)
        deepModel.save('{0}/ModelsMLP/deepMLP{1}.hdf5'.format(self.path, self.dataset))

        if plots:
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(np.arange(0, self.epochs), deepH.history["loss"], label="train_loss")
            plt.plot(np.arange(0, self.epochs), deepH.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, self.epochs), deepH.history["acc"], label="train_acc")
            plt.plot(np.arange(0, self.epochs), deepH.history["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy on " + self.dataset)
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            plt.savefig('{0}/GraphsMLP/deepMLP{1}.png'.format(self.path, self.dataset))

        self.deep_model = deepModel
        return deepH.history["val_acc"][-1]

    @staticmethod
    def _wideMLP(num_classes):
        n_hidden_1 = 200  # 1st layer number of neurons
        n_hidden_2 = 75  # 2nd layer number of neurons

        model = Sequential()
        model.add(Dense(n_hidden_1, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(n_hidden_2, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(num_classes, activation='softmax'))

        return model

    @staticmethod
    def _deepMLP(num_classes):

        n_hidden_1 = 100  # 1st layer number of neurons
        n_hidden_2 = 75  # 2nd layer number of neurons
        n_hidden_3 = 50  # 3rd layer number of neurons
        n_hidden_4 = 30  # 3rd layer number of neurons

        model = Sequential()
        model.add(Dense(n_hidden_1, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(n_hidden_2, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(n_hidden_3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(n_hidden_4, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(num_classes, activation='softmax'))

        return model

    def _load_model(self, type):
        return load_model('{0}/ModelsMLP/{1}MLP{2}.hdf5'.format(self.path, type, self.dataset))

    def load_deep_model(self):
        return self._load_model('deep')

    def load_wide_model(self):
        return self._load_model('wide')

    def predict_wide_model(self):
        y_predict = self.wide_model.predict(self.X_test)
        return y_predict

    def predict_deep_model(self):
        y_predict = self.deep_model.predict(self.X_test)
        return y_predict