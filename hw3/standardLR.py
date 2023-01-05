import argparse
import numpy as np
import pandas as pd
import time
from numpy.linalg import inv

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}


        start=time.time()

        # add dummy column for beta 0
        dummy_ones_train=np.ones(shape=(len(yTrain),1))
        xTrain=np.column_stack((dummy_ones_train,xTrain))

        dummy_ones_test=np.ones(shape=(len(yTest),1))
        xTest=np.column_stack((dummy_ones_test,xTest))

        # close form solution
        weights=inv(xTrain.transpose().dot(xTrain)).dot(xTrain.transpose()).dot(yTrain)
        self.beta=weights

        trainingMSE=self.mse(xTrain,yTrain)
        testingMSE=self.mse(xTest,yTest)

        end=time.time()
        time_elapsed=end-start
        result={"time":time_elapsed,"train-mse":trainingMSE,"test-mse":testingMSE}
        trainStats[0]=result
        return trainStats


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

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
