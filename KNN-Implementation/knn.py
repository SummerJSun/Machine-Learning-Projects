import argparse
import numpy as np
import pandas as pd
'''
import matplotlib.pyplot as plt
import seaborn as sns
'''


class Knn(object):
    k = 0              # number of neighbors to use
    nFeatures = 0      # number of features seen in training
    nSamples = 0       # number of samples seen in training
    isFitted = False  # has train been called on a dataset?


    def __init__(self, k): 
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.x = None
        self.y = None
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : numpy nd-array with shape (n, d)
            Training data 
        y : numpy 1d array with shape (n, )
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        self.x = np.array(xFeat)
        self.y = np.array(y)
        self.isFitted = True
        return self
    

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.
            
        Parameters
        ----------
        xFeat : numpy nd-array with shape (m, d)
            The data to predict.  

        Returns
        -------
        yHat : numpy.1d array with shape (m, )
            Predicted class label per sample
        """


        distance = np.zeros((xFeat.shape[0], self.x.shape[0]))
        yHat = []

        for i in range(len(xFeat)):
            for j in range(len(self.x)):
                distance[i,j] = np.linalg.norm(xFeat[i]-self.x[j])
        
        for dist in distance: 
            nearest = np.argsort(dist)[:self.k]
            nearest_labels = self.y[nearest]
            vote,counts = np.unique(nearest_labels,return_counts=True)
            common = vote[counts.argmax()]
            yHat.append(common)
        
        return np.array(yHat)


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape (n,)
        Predicted class label for n samples
    yTrue : 1d-array with shape (n, )
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    correct = np.sum(yHat == yTrue)
    total = len (yHat)
    acc = correct/total
    return acc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="simxTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="simyTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="simxTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="simyTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    # assume the data is all numerical and 
    # no additional pre-processing is necessary
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain)
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest)
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

    
    #for question d
    '''
    k_val = range(1,50)
    train_acu = []
    test_acu = []
    for k in k_val:
        knn = Knn(k)
        knn.train(xTrain, yTrain)
        Train = knn.predict(xTrain)
        Test = knn.predict(xTest)
        trainacc = accuracy(Train, yTrain)
        testacc = accuracy(Test, yTest)
        train_acu.append(trainacc)
        test_acu.append(testacc)     

    plt.figure(figsize=(12, 6))
    plt.plot(k_val, train_acu, label='Train Accuracy')
    plt.plot(k_val, test_acu, label='Test Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.title('Accuracy(%) vs Number of Neighbors')
    plt.legend()
    plt.show()
    '''
    



if __name__ == "__main__":
    main()
