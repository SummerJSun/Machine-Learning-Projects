import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from preprocess import standard_scale, minmax_range, add_irr_feature
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
'''
from tabulate import tabulate
'''


def evaluate_knn(xTrain, yTrain, xTest, yTest):
    """
    Train a knn using xTrain and yTrain, and then predict
    the labels of the test data. This method will should
    return the classifier itself and the accuracy of the 
    resulting trained model on the test set.

    Parameters
    ----------
    xTrain : numpy nd-array with shape (n, d)
        Training data 
    yTrain : numpy 1d array with shape (n,)
        Array of labels associated with training data.
    xTest : numpy nd-array with shape (m, d)
        Test data 
    yTest : numpy 1d array with shape (m, )
        Array of labels associated with test data.

    Returns
    -------
    knn : an instance of the sklearn classifier associated with knn
        The knn model trained
    acc : float
        The accuracy of the trained model on the test data
    """
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(xTrain, yTrain)
    y_pred = knn.predict(xTest)
    acc = accuracy_score(yTest, y_pred)

    return knn, acc


def evaluate_nb(xTrain, yTrain, xTest, yTest):
    """
    Train a knn using xTrain and yTrain, and then predict
    the labels of the test data. This method will should
    return the classifier itself and the accuracy of the 
    resulting trained model on the test set.

    Parameters
    ----------
    xTrain : numpy nd-array with shape (n, d)
        Training data 
    yTrain : numpy 1d array with shape (n,)
        Array of labels associated with training data.
    xTest : numpy nd-array with shape (m, d)
        Test data 
    yTest : numpy 1d array with shape (m, )
        Array of labels associated with test data.

    Returns
    -------
    knn : an instance of the sklearn classifier associated with knn
        The knn model trained
    acc : float
        The accuracy of the trained model on the test data
    """
    knn = GaussianNB()
    knn.fit(xTrain, yTrain)
    y_pred = knn.predict(xTest)
    acc = accuracy_score(yTest, y_pred)
    return knn, acc


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="space_trainx.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="space_trainy.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="space_testx.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="space_testy.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    # flatten to compress to 1-d rather than (m, 1)
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()

    # add code here
    # 4c
    results = []

    # No preprocessing
    knn_acc = evaluate_knn(xTrain, yTrain, xTest, yTest)
    nb_acc = evaluate_nb(xTrain, yTrain, xTest, yTest)
    results.append(['No Pre-processing', knn_acc, nb_acc])

    # Standard scale
    xTrain_scaled, xTest_scaled = standard_scale(xTrain, xTest)
    knn_acc = evaluate_knn(xTrain_scaled, yTrain, xTest_scaled, yTest)
    nb_acc = evaluate_nb(xTrain_scaled, yTrain, xTest_scaled, yTest)
    results.append(['Standard_Scale', knn_acc, nb_acc])

    # MinMax scale
    xTrain_minmax, xTest_minmax = minmax_range(xTrain, xTest)
    knn_acc = evaluate_knn(xTrain_minmax, yTrain, xTest_minmax, yTest)
    nb_acc = evaluate_nb(xTrain_minmax, yTrain, xTest_minmax, yTest)
    results.append(['MinMax_Scale', knn_acc, nb_acc])

    # Add irrelevant feature
    xTrain_irr, xTest_irr = add_irr_feature(xTrain, xTest)
    knn_acc = evaluate_knn(xTrain_irr, yTrain, xTest_irr, yTest)
    nb_acc = evaluate_nb(xTrain_irr, yTrain, xTest_irr, yTest)
    results.append(['add_irr_feature', knn_acc, nb_acc])

    '''
    #creating table
    # Creating table
    result = [{'Preprocessing': r[0], 'KNN Accuracy': r[1][1], 'NB Accuracy': r[2][1]} for r in results]
    print(tabulate(result, headers="keys", tablefmt="grid"))
    '''



if __name__ == "__main__":
    main()
