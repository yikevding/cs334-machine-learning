import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rf import RandomForest
from rf import accuracy
import matplotlib.pyplot as plt
from rf import errors




def main():
    xTrain = pd.read_csv("q4xTrain.csv")
    yTrain = pd.read_csv("q4yTrain.csv")
    xTest = pd.read_csv("q4xTest.csv")
    yTest = pd.read_csv("q4yTest.csv")

    # information gained from part b
    optimal_nest=20
    optimal_depth=7
    optimal_leaf_samples=45
    optimal_features=3
    optimal_criterion="gini"

    # using the optimal parameters
    model=RandomForest(nest=optimal_nest,maxFeat=optimal_features,
                       criterion=optimal_criterion,maxDepth=optimal_depth,minLeafSample=optimal_leaf_samples)
    stats=model.train(xTrain,yTrain)
    oob_cases=model.trees[optimal_nest-1][4]
    oob_errors=stats[optimal_nest]
    print("There are",oob_errors,"estimated OOB error out of",oob_cases,"out of bag cases.")
    train_accuracy=(oob_cases-oob_errors)/oob_cases
    print("Train accuracy is",train_accuracy)
    yHat=model.predict(xTest)
    test_errors=errors(yTest,yHat)
    test_accuracy=accuracy(yTest,yHat)
    total=len(yTest)
    print("There are",test_errors,"misclassification errors out of",total,"test cases.")
    print("Test accuracy is",test_accuracy)


if __name__ == "__main__":
    main()