import argparse
import numpy as np
import pandas as pd
import time

class Perceptron():
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

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

        # TODO implement this

        # initialize weights with the dummy 1 in the front
        self.w=np.ones(shape=(xFeat[0].size+1,))

        # perceptron
        for epoch in range(1,self.mEpoch+1):
            mistakes=0
            index=0
            for row in xFeat:
                array=np.insert(row,0,1)
                value=np.dot(self.w,array)
                truth=y[index]
                if(value>=0):
                    predicted=1
                else:
                    predicted=0
                if(predicted!=truth):
                    mistakes+=1
                    # mistake on positive
                    if(predicted==0 and truth==1):
                        self.w=self.w+array
                    # mistake on negative
                    elif(predicted==1 and truth ==0):
                        self.w=self.w-array
                index+=1
            if(mistakes==0):
                break
            stats[epoch]=mistakes

        return stats

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """
        yHat = []
        for row in xFeat:
            array = np.insert(row, 0, 1)
            value = np.dot(self.w, array)
            if(value>=0):
                yHat.append(1)
            else:
                yHat.append(0)
        return yHat


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    ans=0
    index=0
    for value in yTrue:
        if(yHat[index]!=value):
            ans+=1
        index+=1
    return ans


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()

    # load the train and test data assumes you'll use numpy
    xTrain = np.loadtxt(args.xTrain,delimiter=",")
    yTrain = np.loadtxt(args.yTrain,delimiter=",")
    xTest = np.loadtxt(args.xTest,delimiter=",")
    yTest = np.loadtxt(args.yTest,delimiter=",")

    np.random.seed(args.seed)   
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))


if __name__ == "__main__":
    main()
