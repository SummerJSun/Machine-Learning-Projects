import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class Perceptron(object):
    mEpoch = 0       # maximum epoch size
    intercept = None # model the intercept - if false assume no intercept 
    w = None         # weights of the perceptron

    def __init__(self, epoch, intercept=True):
        self.mEpoch = epoch
        self.w = None
        self.intercept = intercept

    def train(self, xFeat, y):
        """
        Train the perceptron using the data.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}  
        if self.intercept:
            self.w = np.zeros(xFeat.shape[1]+1)
            xFeat = pad_x(xFeat)
        else:
            self.w = np.zeros(xFeat.shape[1])
        for epoch in range(1, self.mEpoch + 1): 
            mistakes = 0
            # shuffle
            indices = np.random.permutation(len(xFeat))
            xFeat = xFeat[indices]
            y = y[indices]  
            for xi, yi in zip(xFeat, y):
                mistake = self.sample_update(xi, yi)
                mistakes += mistake
            stats[epoch] = mistakes
            if mistakes == 0:
                break
        return stats

    def sample_update(self, xi, yi):
        """
        Given a single sample, update the perceptron weight
        and return whether or not there was a mistake.
        xi is assumed to be padded and the appropriate
        dimension has been set.

        Parameters
        ----------
        xi : numpy array of shape (1, d+1)
            Training sample 
        y : single value (-1, +1)
            Training label

        Returns
        -------
            mistake: 0/1 
                Was there a mistake made 
        """
        prediction = np.sign(np.dot(self.w, xi.T))
        if prediction != yi:
            self.w += yi * xi
            return 1
        else:
            return 0
        
    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape (m, d)
            The data to predict.  

        Returns
        -------
        yHat : numpy.1d array of shape (m, )
            Predicted response per sample
        """
        if self.intercept:
            xFeat = pad_x(xFeat)
        yHat = np.sign(np.dot(xFeat, self.w))
        yHat[yHat == 0] = 1
        return yHat
    


def transform_y(y):
    """
    Given a numpy 1D array with 0/1, transform the y 
    label to be -1/1

    Parameters
    ----------
    y : numpy 1-d array with labels of 0 and 1
        The true label.      

    Returns
    -------
    y : numpy 1-d array with labels of -1 and 1
        where the original 0 -> -1
    """
    #transfer lable 0/1 to -1/1
    y[y == 0] = -1
    return y

def pad_x(xMat):
    """
    Given the feature matrix xMat, add padding
    (i.e., column of 1s) to either the beginning
    or end of the feature matrix.

    Parameters
    ----------
    xMat : nd-array with shape (n, d)
        The feature matrix.  

    Returns
    -------
    xPad : nd-array with shape (n, d+1)
        Padded feature matrix
    """
    xPad = np.hstack((xMat, np.ones((xMat.shape[0], 1))))
    return xPad


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : numpy 1-d array shape (n, )
        The predicted label.
    yTrue : numpy 1-d array shape (n, )
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    err = np.sum(yHat != yTrue)
    return err


def tune_perceptron(trainx, trainy, epochList):
    """
    Tune the preceptron using k-fold CV to find the 
    optimal number of epochs

    Parameters
    ----------
    trainx : a nxd numpy array
    trainy : numpy 1d array of shape n
        The true label.    
    epochList: a list of positive integers
        The epoch list to search over  

    Returns
    -------
    epoch : int
        The optimal number of epochs
    """
    error_rates = []
    for epoch in epochList:
        model = Perceptron(epoch)
        error_sum = 0
        for train_index, val_index in KFold(n_splits=5).split(trainx):
            xTrain, xVal = trainx[train_index], trainx[val_index]
            yTrain, yVal = trainy[train_index], trainy[val_index]
            model.train(xTrain, yTrain)
            yValHat = model.predict(xVal)
            error_sum += calc_mistakes(yValHat, yVal)
        error_rate = error_sum / len(trainx)
        error_rates.append(error_rate)
    optimal_epoch = epochList[np.argmin(error_rates)]
    return optimal_epoch

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("epoch", type=int, help="max number of epochs")
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
    # load the train and test data assumes you'll use numpy
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()

    # transform the y
    yTrain = transform_y(yTrain)
    yTest = transform_y(yTest)

    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))
    epochList = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    result = tune_perceptron(xTrain, yTrain, epochList)
    print("Optimal number of epochs: ", result)


if __name__ == "__main__":
    main()