import argparse
import numpy as np
import pandas as pd
import time
from lr import LinearRegression, file_to_numpy


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SGD

        start=time.time()

        # add dummy column for beta 0
        dummy_ones_train = np.ones(shape=(len(yTrain), 1))
        xTrain = np.column_stack((dummy_ones_train, xTrain))

        dummy_ones_test = np.ones(shape=(len(yTest), 1))
        xTest = np.column_stack((dummy_ones_test, xTest))

        # randomly initializa beta to be 1
        self.beta = np.ones(shape=(len(xTrain[0]),))

        train_combo=np.column_stack((xTrain,yTrain))
        iteration=0
        for epoch in range(1,self.mEpoch+1):
            print(epoch)
            B=len(train_combo)/self.bs
            np.random.shuffle(train_combo)
            batches=np.array_split(train_combo,B)
            # for each batch, compute average gradient
            for batch in batches:
                gradient_sum=0
                for row in batch:
                    label=row[len(row)-1]
                    xs=row[0:len(row)-1]
                    gradient_sum+=xs.transpose().dot(xs.dot(self.beta)-label)
                gradient=gradient_sum/len(batch)
                self.beta=self.beta-self.lr*gradient
                trainingMSE=self.mse(xTrain,yTrain)
                testingMSE=self.mse(xTest,yTest)
                end = time.time()
                elapsed = end - start
                result={"time":elapsed,"train-mse":trainingMSE,"test-mse":testingMSE}
                trainStats[iteration]=result
                iteration+=1

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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)   
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()

