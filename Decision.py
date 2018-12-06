import numpy as np
import csv
import pandas as pd


def MLPdecision(categories, methode):
    predict1 = pd.read_csv('ResultatsCSV\\MLP_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 0:24]
    predict2 = pd.read_csv('ResultatsCSV\\MLP_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 25:49]
    predict3 = pd.read_csv('ResultatsCSV\\MLP_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 50:74]
    IDs = pd.read_csv('ResultatsCSV\\MLP_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 75]
    if methode=='sum':
        sum_predict = np.add(predict1, np.add(predict2, predict3))
    if methode=='max':
        max_predict = np.maximum(predict1, np.maximum(predict2, predict3))
    if methode=='min':
        min_predict = np.minimum(predict1, np.minimum(predict2, predict3))
    if methode in ('sum', 'max', 'min'):
        with open('ResultatsCSV\\MLP_decision_unlabeled.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            filewriter.writerow(['id', 'genre'])
            for i in range(len(IDs)):
                filewriter.writerow([IDs[i], categories[np.argmax(sum_predict[i])]])


def LDAdecision(categories, methode):
    predict1 = pd.read_csv('ResultatsCSV\\LDA_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 0:24]
    predict2 = pd.read_csv('ResultatsCSV\\LDA_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 25:49]
    predict3 = pd.read_csv('ResultatsCSV\\LDA_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 50:74]
    IDs = pd.read_csv('ResultatsCSV\\LDA_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 75]
    if methode=='sum':
        sum_predict = np.add(predict1, np.add(predict2, predict3))
    if methode=='max':
        max_predict = np.maximum(predict1, np.maximum(predict2, predict3))
    if methode=='min':
        min_predict = np.minimum(predict1, np.minimum(predict2, predict3))
    if methode in ('sum', 'max', 'min'):
        with open('ResultatsCSV\\LDA_decision_unlabeled.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            filewriter.writerow(['id', 'genre'])
            for i in range(len(IDs)):
                filewriter.writerow([IDs[i], categories[np.argmax(sum_predict[i])]])


def SVMdecision(categories, methode):
    predict1 = pd.read_csv('ResultatsCSV\\SVM_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 0:24]
    predict2 = pd.read_csv('ResultatsCSV\\SVM_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 25:49]
    predict3 = pd.read_csv('ResultatsCSV\\SVM_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 50:74]
    IDs = pd.read_csv('ResultatsCSV\\SVM_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 75]
    if methode=='sum':
        sum_predict = np.add(predict1, np.add(predict2, predict3))
    if methode=='max':
        max_predict = np.maximum(predict1, np.maximum(predict2, predict3))
    if methode=='min':
        min_predict = np.minimum(predict1, np.minimum(predict2, predict3))
    if methode in ('sum', 'max', 'min'):
        with open('ResultatsCSV\\SVM_decision_unlabeled.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            filewriter.writerow(['id', 'genre'])
            for i in range(len(IDs)):
                filewriter.writerow([IDs[i], categories[np.argmax(sum_predict[i])]])


def totalDecision(categories, methode):
    predict1 = pd.read_csv('ResultatsCSV\\SVM_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 0:24]
    predict2 = pd.read_csv('ResultatsCSV\\SVM_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 25:49]
    predict3 = pd.read_csv('ResultatsCSV\\SVM_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 50:74]
    predict4 = pd.read_csv('ResultatsCSV\\LDA_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 0:24]
    predict5 = pd.read_csv('ResultatsCSV\\LDA_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 25:49]
    predict6 = pd.read_csv('ResultatsCSV\\LDA_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 50:74]
    predict7 = pd.read_csv('ResultatsCSV\\MLP_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 0:24]
    predict8 = pd.read_csv('ResultatsCSV\\MLP_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 25:49]
    predict9 = pd.read_csv('ResultatsCSV\\MLP_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 50:74]
    IDs = pd.read_csv('ResultatsCSV\\SVM_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 75]
    if methode=='sum':
        sum_predict = np.add(predict1, np.add(predict2, np.add(predict3, np.add(predict4, np.add(predict5, np.add(predict6,
                             np.add(predict7, np.add(predict8, predict9))))))))
    if methode=='max':
        max_predict = np.maximum(predict1, np.maximum(predict2, np.maximum(predict3, np.maximum(predict4, np.maximum(predict5, np.maximum(predict6,
                             np.maximum(predict7, np.maximum(predict8, predict9))))))))
    if methode=='min':
        min_predict = np.minimum(predict1, np.minimum(predict2, np.minimum(predict3, np.minimum(predict4, np.minimum(predict5, np.minimum(predict6,
                             np.minimum(predict7, np.minimum(predict8, predict9))))))))
    if methode in ('sum', 'max', 'min'):
        with open('ResultatsCSV\\total_decision_unlabeled.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            filewriter.writerow(['id', 'genre'])
            for i in range(len(IDs)):
                filewriter.writerow([IDs[i], categories[np.argmax(sum_predict[i])]])


categories = ('BIG_BAND', 'BLUES_CONTEMPORARY', 'COUNTRY_TRADITIONAL', 'DANCE', 'ELECTRONICA', 'EXPERIMENTAL',
              'FOLK_INTERNATIONAL', 'GOSPEL', 'GRUNGE_EMO', 'HIP_HOP_RAP', 'JAZZ_CLASSIC', 'METAL_ALTERNATIVE',
              'METAL_DEATH', 'METAL_HEAVY', 'POP_CONTEMPORARY', 'POP_INDIE', 'POP_LATIN', 'PUNK', 'REGGAE',
              'RNB_SOUL', 'ROCK_ALTERNATIVE', 'ROCK_COLLEGE', 'ROCK_CONTEMPORARY', 'ROCK_HARD', 'ROCK_NEO_PSYCHEDELIA')
MLPdecision(categories, 'sum')
LDAdecision(categories, 'sum')
SVMdecision(categories, 'sum')
totalDecision(categories, 'sum')
