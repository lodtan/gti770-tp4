import numpy as np
import csv
import pandas as pd
from Datasets import CATEGORIES

class Voting(object):

    ''' Instantiate MLP with Keras'''

    def __init__(self,path):

        print("[INFO] Starting decision module...")

        self.path = path


    def MLPdecision(self,categories, methode):

        print("[INFO] openning MLP predictions...")

        predict = pd.read_csv('{0}/MLP_Predictions_unlabeled.csv'.format(self.path), delimiter=',', header=None)
        predict1 = predict.values[:, 0:24]
        predict2 = predict.values[:, 25:49]
        predict3 = predict.values[:, 50:74]

        IDs = predict.values[:, 75]
        if methode == 'sum':
            decision = np.add(predict1, np.add(predict2, predict3))
        if methode == 'max':
            decision = np.maximum(predict1, np.maximum(predict2, predict3))
        if methode == 'min':
            decision = np.minimum(predict1, np.minimum(predict2, predict3))
        if methode in ('sum', 'max', 'min'):
            with open('{0}/MLP_Predictions_voting_unlabeled.csv'.format(self.path), 'w', newline='') as csvfile:
                print("[INFO] writting MLP decisions...")
                filewriter = csv.writer(csvfile, delimiter=',')
                filewriter.writerow(['id', 'genre'])
                for i in range(len(IDs)):
                    filewriter.writerow([IDs[i], categories[np.argmax(decision[i])]])
        return decision


    def LDAdecision(self,categories, methode):

        print("[INFO] openning LDA predictions...")
        predict1 = pd.read_csv('ResultatsCSV\\LDA_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 0:24]
        predict2 = pd.read_csv('ResultatsCSV\\LDA_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 25:49]
        predict3 = pd.read_csv('ResultatsCSV\\LDA_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 50:74]
        IDs = pd.read_csv('ResultatsCSV\\LDA_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 75]
        if methode == 'sum':
            decision = np.add(predict1, np.add(predict2, predict3))
        if methode == 'max':
            decision = np.maximum(predict1, np.maximum(predict2, predict3))
        if methode == 'min':
            decision = np.minimum(predict1, np.minimum(predict2, predict3))
        if methode in ('sum', 'max', 'min'):
            with open('ResultatsCSV\\LDA_decision_unlabeled.csv', 'w', newline='') as csvfile:
                print("[INFO] writting LDA decisions...")
                filewriter = csv.writer(csvfile, delimiter=',')
                filewriter.writerow(['id', 'genre'])
                for i in range(len(IDs)):
                    filewriter.writerow([IDs[i], categories[np.argmax(decision[i])]])


    def SVMdecision(self,categories, methode):

        print("[INFO] openning SVM predictions...")
        predict1 = pd.read_csv('ResultatsCSV\\SVM_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 0:24]
        predict2 = pd.read_csv('ResultatsCSV\\SVM_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 25:49]
        predict3 = pd.read_csv('ResultatsCSV\\SVM_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 50:74]
        IDs = pd.read_csv('ResultatsCSV\\SVM_Predictions_unlabeled.csv', delimiter=',', header=None).values[:, 75]
        if methode == 'sum':
            decision = np.add(predict1, np.add(predict2, predict3))
        if methode == 'max':
            decision = np.maximum(predict1, np.maximum(predict2, predict3))
        if methode == 'min':
            decision = np.minimum(predict1, np.minimum(predict2, predict3))
        if methode in ('sum', 'max', 'min'):
            with open('ResultatsCSV\\SVM_decision_unlabeled.csv', 'w', newline='') as csvfile:
                print("[INFO] writting SVM decisions...")
                filewriter = csv.writer(csvfile, delimiter=',')
                filewriter.writerow(['id', 'genre'])
                for i in range(len(IDs)):
                    filewriter.writerow([IDs[i], categories[np.argmax(decision[i])]])


    def totalDecision(self,categories, methode):

        print("[INFO] openning SVM, LDA and MLP predictions...")
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
        if methode == 'sum':
            decision = np.add(predict1, np.add(predict2, np.add(predict3, np.add(predict4, np.add(predict5, np.add(predict6,
                                 np.add(predict7, np.add(predict8, predict9))))))))
        if methode == 'max':
            decision = np.maximum(predict1, np.maximum(predict2, np.maximum(predict3, np.maximum(predict4, np.maximum(predict5, np.maximum(predict6,
                                 np.maximum(predict7, np.maximum(predict8, predict9))))))))
        if methode == 'min':
            decision = np.minimum(predict1, np.minimum(predict2, np.minimum(predict3, np.minimum(predict4, np.minimum(predict5, np.minimum(predict6,
                                 np.minimum(predict7, np.minimum(predict8, predict9))))))))
        if methode in ('sum', 'max', 'min'):
            with open('ResultatsCSV\\total_decision_unlabeled.csv', 'w', newline='') as csvfile:
                print("[INFO] writting globale decisions...")
                filewriter = csv.writer(csvfile, delimiter=',')
                filewriter.writerow(['id', 'genre'])
                for i in range(len(IDs)):
                    filewriter.writerow([IDs[i], categories[np.argmax(decision[i])]])


#if __name__ == '__main__':
#    MLPdecision(CATEGORIES, STRATEGIES[0])
#    LDAdecision(CATEGORIES, STRATEGIES[0])
    #SVMdecision(CATEGORIES, STRATEGIES[0])
    #totalDecision(CATEGORIES, STRATEGIES[0])
