import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.trees=[]
        self.features=maxFeat

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """

        stats={}
        # construct forest
        size=len(xFeat)
        for i in range(0,self.nest):
            # for each tree select subset of features
            errors=0
            subspace=xFeat.sample(n=self.features,axis='columns')
            # store column name
            columns=[]
            for column in subspace:
                columns.append(column)
            # for each tree boostrap some samples
            bootstrapped=subspace.sample(size,replace=True)
            used_list=[]
            labels=[]
            for index,row in bootstrapped.iterrows():
                used_list.append(index)
                labels.append(y["label"][index])
            used=set(used_list)
            model=DecisionTreeClassifier(criterion=self.criterion,max_depth=self.maxDepth,min_samples_leaf=self.minLeafSample)
            model.fit(bootstrapped,labels)
            oob_samples=len(subspace)-len(used) # number of oob samples
            self.trees.append([model,used,subspace,columns,oob_samples])

            # now calculate the oob error until this point
            for num,data in subspace.iterrows():
                zeros=0
                ones=0
                current_values=model.predict(subspace)
                if(num not in used):
                    current=current_values[num]
                    if(current==0):
                        zeros+=1
                    else:
                        ones+=1
                    for id in range(0,i): # to check the previous trees
                        tree = self.trees[id][0]
                        inbag = self.trees[id][1]
                        dataset = self.trees[id][2]
                        values = tree.predict(dataset)
                        if(num not in inbag):
                            value=values[num]
                            if(value==0):
                                zeros+=1
                            else:
                                ones+=1
                    predicted=0
                    if(ones>zeros):
                        predicted=1
                    if(ones==zeros):
                        predicted=np.random.randint(2,size=1)[0]
                    true=y["label"][num]
                    if(predicted!=true):
                        errors+=1
            stats[i+1]=errors
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
        predict_lists=[]
        for tree in self.trees:
            model=tree[0]
            columns=tree[3]
            testing=pd.DataFrame()
            for column in columns:
                testing[column]=xFeat[column]
            labels=model.predict(testing)
            predict_lists.append(labels)

        for index,row in xFeat.iterrows():
            zeros=0
            ones=0
            predicted=0
            for i in range(0,self.nest):
                value=predict_lists[i][index]
                if(value==0):
                    zeros+=1
                else:
                    ones+=1
            if(ones>zeros):
                predicted=1
            if(ones==zeros):
                predicted = np.random.randint(2, size=1)[0]
            yHat.append(predicted)
        return yHat


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def accuracy(yTrue,yHat):
    errors=0
    for index in range(0,len(yTrue)):
        if(yTrue["label"][index]!=yHat[index]):
            errors+=1

    return (len(yTrue)-errors)/len(yTrue)

def errors(yTrue,yHat):
    errors=0
    for index in range(0,len(yTrue)):
        if(yTrue["label"][index]!=yHat[index]):
            errors+=1
    return errors






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
    parser.add_argument("criterion",help="criteria to determine the split")
    parser.add_argument("maxFeat",help="maximum number of features to consider for each tree")
    parser.add_argument("maxDepth",help="maximum depth of the tree")
    parser.add_argument("minLeafSample",help="minimum number of samples in leaf node")
    parser.add_argument("nest",help="number of trees")
    # parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed",default=334,
                        type=int, help="default seed number")
    
    args = parser.parse_args()

    # load the train and test data assumes you'll use numpy
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    np.random.seed(args.seed)
    nest=int(args.nest)
    maxFeat=int(args.maxFeat)
    maxDepth=int(args.maxDepth)
    minLeafSample=int(args.minLeafSample)
    model = RandomForest(nest,maxFeat,args.criterion,maxDepth,minLeafSample)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    print("Accuracy: ",accuracy(yTest,yHat))


if __name__ == "__main__":
    main()