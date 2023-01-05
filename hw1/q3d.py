import knn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    max_depth=10
    values=list(range(1,max_depth+1))
    testAccuracy=[]
    trainAccuracy=[]

    ## load datasets
    xTrain=pd.read_csv("q3xTrain.csv")
    yTrain=pd.read_csv("q3yTrain.csv")
    xTest=pd.read_csv("q3xTest.csv")
    yTest=pd.read_csv("q3yTest.csv")

    ## fit and predict
    for k in range(1,max+1):
        model=knn.Knn(k)
        model.train(xTrain,yTrain["label"])
        yHatTest=model.predict(xTest)
        testRate=knn.accuracy(yHatTest,yTest["label"])
        testAccuracy.append(testRate)
        yHatTrain=model.predict(xTrain)
        trainRate=knn.accuracy(yHatTrain,yTest["label"])
        trainAccuracy.append(trainRate)

    ## plot graph
    plt.plot(values,trainAccuracy,color="red",label="Training")
    plt.plot(values,testAccuracy,color="blue",label="Testing")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Accuracy with different k values")
    plt.legend()
    plt.savefig("3d.png")
    plt.show()


if __name__ == "__main__":
    main()

