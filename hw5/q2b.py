import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rf import RandomForest
import matplotlib.pyplot as plt



def main():
    xTrain=pd.read_csv("q4xTrain.csv")
    yTrain=pd.read_csv("q4yTrain.csv")

    model=RandomForest(nest=100,maxFeat=5,criterion="gini",maxDepth=5,minLeafSample=30)
    stats=model.train(xTrain,yTrain)
    Xs=[]
    errors=[]
    for key in stats:
        Xs.append(key)
        errors.append(stats[key])

    # plot a line graph to see the trend of the oob error
    plt.plot(Xs,errors)
    plt.xlabel("number of trees")
    plt.ylabel("out of bag errors")
    plt.savefig("oob.png")
    plt.show()

    # from the plot we know that the best nest is somewhere near nest=25
    # now do grid search to tune the other hyperparameters

    # find the best features, between 2-8
    grid={}
    max=8
    Xs=[]
    errors=[]
    for feature in range(2,max+1):
        model=RandomForest(nest=5,maxFeat=feature,criterion="gini",maxDepth=5,minLeafSample=30)
        stats=model.train(xTrain,yTrain)
        grid[feature]=stats[5]
        Xs.append(feature)
        errors.append(stats[5])
    optimal_features=2
    oob=grid[2]
    for key in grid:
        if(grid[key]<oob):
            oob=grid[key]
            optimal_features=key
    print("Optimal number of features chosen is",optimal_features)
    plt.plot(Xs,errors)
    plt.ylabel("out of bag errors")
    plt.xlabel("number of features")
    plt.savefig("features.png")
    plt.show()

    # find best criterion, between gini and entropy
    optimal_criterion="gini"
    model_gini=RandomForest(nest=5,maxFeat=6,criterion="gini",maxDepth=5,minLeafSample=30)
    stats_gini=model_gini.train(xTrain,yTrain)
    oob_gini=stats_gini[5]
    model_entropy=RandomForest(nest=5,maxFeat=6,criterion="entropy",maxDepth=5,minLeafSample=30)
    stats_entropy=model_entropy.train(xTrain,yTrain)
    oob_entropy=stats_entropy[5]
    if(oob_entropy<=oob_gini):
        optimal_criterion="entropy"
    print("Optimal criterion is",optimal_criterion)

    # find best depth, range between 5-10
    Xs=[]
    errors=[]
    grid={}
    max=10
    for depth in range(5,max+1):
        model=RandomForest(nest=5,maxFeat=6,criterion="entropy",maxDepth=depth,minLeafSample=30)
        stats = model.train(xTrain, yTrain)
        grid[depth] = stats[5]
        Xs.append(depth)
        errors.append(stats[5])
    optimal_depth=5
    oob=grid[5]
    for key in grid:
        if(grid[key]<oob):
            oob=grid[key]
            optimal_depth=key
    print("Optimal max depth is",optimal_depth)
    plt.plot(Xs, errors)
    plt.ylabel("out of bag errors")
    plt.xlabel("maximum depth")
    plt.savefig("depth.png")
    plt.show()

    # find best number of samples in leaf, range between 20-70 gap=5
    Xs=[]
    errors=[]
    grid={}
    max=100
    for num in range(20,max+1,5):
        model = RandomForest(nest=5, maxFeat=6, criterion="entropy", maxDepth=7, minLeafSample=num)
        stats = model.train(xTrain, yTrain)
        grid[num] = stats[5]
        Xs.append(num)
        errors.append(stats[5])
    optimal_num = 20
    oob = grid[20]
    for key in grid:
        if (grid[key] < oob):
            oob = grid[key]
            optimal_num = key
    print("Optimal samples in leaf node is", optimal_num)
    plt.plot(Xs, errors)
    plt.ylabel("out of bag errors")
    plt.xlabel("number of samples in leaf node")
    plt.savefig("leaf.png")
    plt.show()


if __name__ == "__main__":
    main()