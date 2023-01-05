
import pandas as pd
from sklearn.model_selection import train_test_split
from sgdLR import SgdLR
import numpy as np
import matplotlib.pyplot as plt







def main():

    # load datasets
    xTrain=pd.read_csv("new_xTrain.csv")
    yTrain=pd.read_csv("eng_yTrain.csv")

    xTrain,xTest,yTrain,yTest=train_test_split(xTrain,yTrain,test_size=0.6) # we use 40% of the original training data
    xTrain=xTrain.to_numpy()
    xTest=xTest.to_numpy()
    yTrain=yTrain.to_numpy()
    yTest=yTest.to_numpy()

    size=len(xTrain)
    rates=[0.01,0.001,0.005,0.0001,0.00001]
    max_epoch = 30
    epochs=list(range(1,max_epoch+1))
    mses=[]
    for rate in rates:
        mse=[]
        bs=1
        model=SgdLR(rate,bs,max_epoch)
        result=model.train_predict(xTrain,yTrain,xTest,yTest)
        for key in range(size-1,max_epoch*size,size):
            mse.append(result[key]["train-mse"])
        mses.append(mse)

    plt.plot(epochs,mses[0],color="red",label="lr=0.01")
    plt.plot(epochs,mses[1],color="blue",label="lr=0.001")
    plt.plot(epochs,mses[2],color="yellow",label="lr=0.005")
    plt.plot(epochs,mses[3],color="green",label="lr=0.0001")
    plt.plot(epochs,mses[4],color="purple",label="lr=0.00001")
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.legend()
    plt.savefig("3b.png")
    plt.show()


if __name__ == "__main__":
    main()