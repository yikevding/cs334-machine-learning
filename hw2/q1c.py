import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dt





def main():
    # data loading
    xTrain=pd.read_csv("q4xTrain.csv")
    yTrain=pd.read_csv("q4yTrain.csv")
    xTest=pd.read_csv("q4xTest.csv")
    yTest=pd.read_csv("q4yTest.csv")


    # get data of switching minimum leaf size
    max_sample=80
    default_depth=6
    xs1=list(range(1,max_sample+1,2))
    trainAcccuracy1=[]
    testAccuracy1=[]

    for size in xs1:
        model=dt.DecisionTree("entropy",default_depth,size)
        train,test=dt.dt_train_test(model,xTrain,yTrain,xTest,yTest)
        trainAcccuracy1.append(train)
        testAccuracy1.append(test)


    # get data of switching depth
    max_depth=30
    default_sample=5
    xs2=list(range(1,max_depth+1))
    trainAcccuracy2=[]
    testAccuracy2=[]

    for depth in xs2:
        model=dt.DecisionTree("entropy",depth,default_sample)
        train,test=dt.dt_train_test(model,xTrain,yTrain,xTest,yTest)
        trainAcccuracy2.append(train)
        testAccuracy2.append(test)

    # plot section
    # varying sample size
    plt.plot(xs1,trainAcccuracy1,color="red",label="Training")
    plt.plot(xs1,testAccuracy1,color="blue",label="Testing")
    plt.xlabel("Minimum leaf size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of decision tree prediction with various minimum leaf size")
    plt.legend()
    plt.savefig("1csize.png")
    plt.show()

    # varying max depth
    plt.plot(xs2, trainAcccuracy2, color="red", label="Training")
    plt.plot(xs2, testAccuracy2, color="blue", label="Testing")
    plt.xlabel("Max depth")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of decision tree with various max depth")
    plt.legend()
    plt.savefig("1cdepth.png")
    plt.show()



















if __name__ == "__main__":
    main()