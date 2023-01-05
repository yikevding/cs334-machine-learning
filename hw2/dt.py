import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import math


class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample




    def build_tree(self,data,depth):
        ## find majority element if reach maximum depth
        if(depth==self.maxDepth):
            labels=data["label"].to_numpy()
            frequency=np.bincount(labels)
            prediction=np.argmax(frequency)
            return {
                "leaf":True,
                "prediction":prediction
            }

        ## loop over each feature
        optimal_feature=""
        optimal_value=0
        mls=True
        gini=1
        entro=2
        for column in data:
            if(column=="label"):
                continue
            values=data[column].to_numpy()
            ## loop over each split value in a feature
            for index, value in enumerate(values):
                # skip over samples that do not satisfy minimum sample size requirements
                left_data = data[(data[column] <= value)]
                left_arr = left_data["label"].to_numpy()
                right_data = data[(data[column] > value)]
                right_arr = right_data["label"].to_numpy()
                parent = data["label"].to_numpy()
                size = parent.size
                if(left_arr.size<self.minLeafSample or right_arr.size<self.minLeafSample):
                    continue
                else:
                    mls=False
                    left_size = left_arr.size
                    right_size = right_arr.size
                    if(self.criterion=="gini"):
                        current=(1.0*left_size/size)*self.gini(left_arr)+(1.0*right_size/size)*self.gini(right_arr)
                        if(current<gini):
                            gini=current
                            optimal_feature=column
                            optimal_value=value
                    elif(self.criterion=="entropy"):
                        current=(1.0*left_size/size)*self.entropy(left_arr)+(1.0*right_size/size)*self.entropy(right_arr)
                        if(current<entro):
                            entro=current
                            optimal_feature=column
                            optimal_value=value

        ## if minimal sample condition is satisfied
        if(mls):
            labels = data["label"].to_numpy()
            frequency = np.bincount(labels)
            prediction = np.argmax(frequency)
            return {
                "leaf": True,
                "prediction": prediction
            }

        ## otherwise slice data and go to left or right child
        left_split=data[(data[optimal_feature]<=optimal_value)]
        right_split=data[(data[optimal_feature]>optimal_value)]
        return {
            "left":self.build_tree(left_split,depth+1),
            "right":self.build_tree(right_split,depth+1),
            "split_feature":optimal_feature,
            "split_value":optimal_value,
            "leaf":False
        }


    # compute gini index
    def gini(self,array):
        negative=(array==0).sum()
        positive=(array==1).sum()
        size=array.size
        neg=1.0*negative/size
        pos=1.0*positive/size
        index=1-(neg)**2-(pos)**2
        return index


    # compute entropy
    def entropy(self,array):
        negative=(array==0).sum()
        positive=(array==1).sum()
        size=array.size
        neg=1.0*negative/size
        pos=1.0*positive/size
        if(neg==0):
            return -pos*math.log(pos,2)
        elif(pos==0):
            return -neg*math.log(neg,2)
        entro=-neg*math.log(neg,2)-pos*math.log(pos,2)
        return entro







    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """

        data=xFeat
        data["label"]=y
        initial_depth=0
        self.tree=self.build_tree(data,initial_depth)

        return self






    def predict_tree(self,tree,row):
        if(tree["leaf"]):
            return tree["prediction"]
        else:
            feature=tree["split_feature"]
            if(row[feature]<=tree["split_value"]):
                return self.predict_tree(tree["left"],row)
            else:
                return self.predict_tree(tree["right"],row)





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
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label
        for index,row in xFeat.iterrows():
            prediction=self.predict_tree(self.tree,row)
            yHat.append(prediction)

        return yHat


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain", default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)



    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
