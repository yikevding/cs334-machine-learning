import pandas as pd
from perceptron import Perceptron
from perceptron import calc_mistakes
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import KFold




def main():

    # load datasets
    xTrain_binary=np.loadtxt("xTrain_binary.csv",delimiter=",")
    xTest_binary=np.loadtxt("xTest_binary.csv",delimiter=",")
    yTrain_binary=np.loadtxt("yTrain_binary.csv",delimiter=",")
    yTest_binary=np.loadtxt("yTest_binary.csv",delimiter=",")

    xTrain_count=np.loadtxt("xTrain_count.csv",delimiter=",")
    xTest_count=np.loadtxt("xTest_count.csv",delimiter=",")
    yTrain_count=np.loadtxt("yTrain_count.csv",delimiter=",")
    yTest_count=np.loadtxt("yTest_count.csv",delimiter=",")


    # binary
    max=20
    epochs=list(range(1,max+1))
    kf = KFold(n_splits=10) # use 10 fold
    stats={}
    for epoch in epochs:
        model=Perceptron(epoch)
        mistakes=[]
        for train_index,test_index in kf.split(xTrain_binary):
            xTrain, xTest = xTrain_binary[train_index], xTrain_binary[test_index]
            yTrain, yTest = yTrain_binary[train_index], yTrain_binary[test_index]
            model.train(xTrain,yTrain)
            yHat=model.predict(xTest)
            mistake=calc_mistakes(yHat,yTest)
            mistakes.append(mistake)
        average=sum(mistakes)/len(mistakes) # average number of mistakes for each epoch
        stats[epoch]=average
    optimal_binary_epoch=0
    min_mistakes=stats[1]
    for key in stats:
        if(stats[key]<min_mistakes):
            min_mistakes=stats[key]
            optimal_binary_epoch=key
    print("Optimal number of epochs for binary set is ",optimal_binary_epoch)

    # count
    max =40
    epochs = list(range(1, max + 1))
    stats={}
    for epoch in epochs:
        model = Perceptron(epoch)
        mistakes = []
        for train_index,test_index in kf.split(xTrain_count):
            xTrain, xTest = xTrain_count[train_index], xTrain_count[test_index]
            yTrain, yTest = yTrain_count[train_index], yTrain_count[test_index]
            model.train(xTrain,yTrain)
            yHat=model.predict(xTest)
            mistake=calc_mistakes(yHat,yTest)
            mistakes.append(mistake)
        average=sum(mistakes)/len(mistakes) # average number of mistakes for each epoch
        stats[epoch]=average
    optimal_count_epoch=0
    min_mistakes=stats[1]
    for key in stats:
        if (stats[key] < min_mistakes):
            min_mistakes = stats[key]
            optimal_count_epoch = key
    print("Optimal number of epochs for count set is ", optimal_count_epoch)


    # binary training and testing
    model=Perceptron(optimal_binary_epoch)
    stats=model.train(xTrain_binary,yTrain_binary)
    yHat=model.predict(xTest_binary)
    train_mistakes=stats[optimal_binary_epoch]
    test_mistakes=calc_mistakes(yHat,yTest_binary)
    print("Using optimal epoch = ",optimal_binary_epoch,"----------")
    print("For binary datasets, number of mistakes on training set is ",train_mistakes)
    print("For binary datasets, number of mistakes on testing set is ",test_mistakes)

    # count training and testing
    model=Perceptron(optimal_count_epoch)
    stats=model.train(xTrain_count,yTrain_count)
    yHat=model.predict(xTest_count)
    train_mistakes=stats[optimal_count_epoch]
    test_mistakes=calc_mistakes(yHat,yTest_count)
    print("Using optimal epoch = ", optimal_count_epoch, "----------")
    print("For count datasets, number of mistakes on training set is ", train_mistakes)
    print("For count datasets, number of mistakes on testing set is ", test_mistakes)








if __name__ == "__main__":
    main()