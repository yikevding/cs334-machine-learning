

import pandas as pd
from sgdLR import SgdLR
import matplotlib.pyplot as plt


def main():
    # load datasets
    xTrain = pd.read_csv("new_xTrain.csv")
    yTrain = pd.read_csv("eng_yTrain.csv")
    xTest=pd.read_csv("new_xTest.csv")
    yTest=pd.read_csv("eng_yTest.csv")

    xTrain=xTrain.to_numpy()
    yTrain=yTrain.to_numpy()
    xTest=xTest.to_numpy()
    yTest=yTest.to_numpy()

    size = len(xTrain)
    lr=0.0001 # obtained from part b
    bs=1
    max_epoch = 30
    epochs = list(range(1, max_epoch + 1))
    train_mse=[]
    test_mse=[]
    model=SgdLR(lr,bs,max_epoch)
    result=model.train_predict(xTrain,yTrain,xTest,yTest)
    for key in range(size-1,max_epoch*size,size):
        train_mse.append(result[key]["train-mse"])
        test_mse.append(result[key]["test-mse"])
    plt.plot(epochs,train_mse,color="red",label="training")
    plt.plot(epochs,test_mse,color="blue",label="testing")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("3c.png")
    plt.show()




if __name__ == "__main__":
    main()