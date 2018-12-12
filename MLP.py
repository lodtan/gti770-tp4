from Datasets import *
from tensorflow.python.client import device_lib
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import BatchNormalization, Dropout
from heapq import nlargest


def deepMLP(num_classes):
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


def wideMLP(num_classes):
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


def createDeepModel(num_classes, X, X_test, Y, Y_test, EPOCHS, BATCH_SIZE, INIT_LR, chosenDataset):
    deepModel = deepMLP(num_classes)
    deepModel.compile(optimizer=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    print("[INFO] training deep model on {0} dataset".format(chosenDataset))
    deepH = deepModel.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                          validation_data=(X_test, Y_test), verbose=0)
    deepModel.save("ModelsMLP\\" + "deepMPL" + chosenDataset + ".hdf5")
    return deepH.history["val_acc"][-1]


def createWideModel(num_classes, X, X_test, Y, Y_test, EPOCHS, BATCH_SIZE, INIT_LR, chosenDataset):
    wideModel = wideMLP(num_classes)
    wideModel.compile(optimizer=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("[INFO] training wide model on {0} dataset".format(chosenDataset))
    wideH = wideModel.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                          validation_data=(X_test, Y_test), verbose = 0)
    wideModel.save("ModelsMLP\\"+"wideMPL"+chosenDataset+".hdf5")
    return wideH.history["val_acc"][-1]


def train(datasets, wide=True, deep=False):
    if not isinstance(datasets, list):
        list(datasets)
    num_classes = len(CATEGORIES)
    BATCH_SIZE = 500
    INIT_LR = 5e-4
    EPOCHS = 50
    deep_val_acc = []
    wide_val_acc = []
    for chosenDataset in datasets:
        X, X_test, Y, Y_test, IDs = loadDataset(chosenDataset)
        X, X_test = scaleDataset(X, X_test)
        if deep:
            deep_val_acc.append(createDeepModel(num_classes, X, X_test, Y, Y_test,
                                                EPOCHS, BATCH_SIZE, INIT_LR, chosenDataset))
            print("[INFO] deep model, last validation accuracy : {0}".format(deep_val_acc[-1]))
        if wide:
            wide_val_acc.append(createWideModel(num_classes, X, X_test, Y, Y_test,
                                                EPOCHS, BATCH_SIZE, INIT_LR, chosenDataset))
            print("[INFO] wide model, last validation accuracy : {0}".format(wide_val_acc[-1]))
    return wide_val_acc, deep_val_acc


def printBestDatasets(Nbests, accuracies):
    max_scores = nlargest(Nbests, accuracies)
    print("Best scores with deep MLP on :\n")
    for i in range(Nbests):
        max_scores_indice = accuracies.index(max_scores[i])
        print("{0} with {1}\n".format(DATASETS[max_scores_indice], max_scores[i]))


if __name__ == '__main__':
    print(device_lib.list_local_devices())
    datasets = (DATASETS[7], DATASETS[4])
    train_acc = train(datasets, wide=False, deep=True)
    printBestDatasets(1, train_acc)
