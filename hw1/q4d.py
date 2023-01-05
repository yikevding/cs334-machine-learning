import knn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import q4

def main():
    max=100
    values = list(range(1, max + 1))
    noPrep = []
    std=[]
    mm=[]
    irr=[]

    ## load datasets
    xTrain = pd.read_csv("q4xTrain.csv")
    yTrain = pd.read_csv("q4yTrain.csv")
    xTest = pd.read_csv("q4xTest.csv")
    yTest = pd.read_csv("q4yTest.csv")

    ## preprocessing
    xTrainStd, xTestStd = q4.standard_scale(xTrain,xTest)

    xTrainMM, xTestMM = q4.minmax_range(xTrain, xTest)

    xTrainIrr, yTrainIrr = q4.add_irr_feature(xTrain, xTest)

    ## fit and predict
    for k in range(1, max + 1):
        model = knn.Knn(k)
        model.train(xTrain, yTrain["label"])
        yHatTest = model.predict(xTest)
        rate = knn.accuracy(yHatTest, yTest["label"])
        noPrep.append(rate)

        model=knn.Knn(k)
        model.train(xTrainStd,yTrain["label"])
        yHatTest=model.predict(xTestStd)
        rate=knn.accuracy(yHatTest,yTest["label"])
        std.append(rate)

        model=knn.Knn(k)
        model.train(xTrainMM,yTrain["label"])
        yHatTest=model.predict(xTestMM)
        rate=knn.accuracy(yHatTest,yTest["label"])
        mm.append(rate)

        model=knn.Knn(k)
        model.train(xTrainIrr,yTrain["label"])
        yHatTest=model.predict(yTrainIrr)
        rate=knn.accuracy(yHatTest,yTest["label"])
        irr.append(rate)

    ## draw graph
    plt.plot(values,noPrep,color="red",label="no-preprocessing")
    plt.plot(values,std,color="blue",label="standard scale")
    plt.plot(values,mm,color="orange",label="min max scale")
    plt.plot(values,irr,color="green",label="irrelevant features")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Accuracy with different k values")
    plt.legend()
    plt.savefig("4d.png")
    plt.show()


if __name__ == "__main__":
    main()