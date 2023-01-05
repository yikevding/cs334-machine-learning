import argparse
import numpy as np
import pandas as pd


class Knn(object):
    k = 0    # number of neighbors to use

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k




    def train(self, xFeat, y):
        """
        Train the k-nn model.

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

        ## check input type
        if(type(xFeat).__module__ == np.__name__):
            self.features=xFeat
            self.labels=y.to_numpy()
        else:
            self.features=xFeat.to_numpy()
            self.labels=y.to_numpy()

        return self



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

        ## first check if input is a numpy array already
        if(not type(xFeat).__module__ == np.__name__):
            xFeat=xFeat.to_numpy()

        ## for each testing point, calculate the distance list and return the top k closest training points to determine simple majority
        for index, data in enumerate(xFeat):
            distances=[]
            for i, value in enumerate(self.features):
                distance=euclidean(data,value)
                distances.append([distance,self.labels[i]])
            arr=np.array(distances)
            arr=arr[arr[:,0].argsort(kind='quicksort')]
            guess=prediction(arr[0:self.k,:])
            yHat.append(guess)

        return yHat



## get the euclidean distance between two vectors
def euclidean(z,x):
    return np.linalg.norm(z-x)


## determine the simple majority
def prediction(array):
    count0=0
    count1=0
    for index,row in enumerate(array):
        if(row[1]==1):
            count1+=1
        else:
            count0+=1
    if(count1>count0):
        return 1
    elif(count0>count1):
        return 0
    else:
        return np.random.randint(2,size=1)[0] # if equal, return a random one




def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """


    acc = 0
    i=0
    while i<len(yHat):
        if(yHat[i]==yTrue[i]):
            acc+=1
        i+=1
    acc=acc/len(yHat)
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
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")



    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)


    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])



    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])


    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)







if __name__ == "__main__":
    main()
