import matplotlib.pyplot as plt
import pandas as pd
from standardLR import StandardLR
from sgdLR import SgdLR
import numpy as np
import matplotlib.pyplot as plt


def main():
    # load datasets
    xTrain=pd.read_csv("new_xTrain.csv").to_numpy()
    xTest=pd.read_csv("new_xTest.csv").to_numpy()
    yTrain=pd.read_csv("eng_yTrain.csv").to_numpy()
    yTest=pd.read_csv("eng_yTest.csv").to_numpy()

    standard_lr =StandardLR()
    record=standard_lr.train_predict(xTrain,yTrain,xTest,yTest)
    train_error=[record[0]["train-mse"]]
    test_error=[record[0]["test-mse"]]
    time_cost=[record[0]["time"]]

    size=len(xTrain)
    batch_sizes=[1,10,430,1290,size]
    max_epoch=2
    learning_rates=[0.0001,0.001,0.005,0.01,0.5]
    times=[]
    train=[]
    test=[]
    for index in range(0,len(batch_sizes)):
        bs=batch_sizes[index]
        lr=learning_rates[index]
        model=SgdLR(lr,bs,max_epoch)
        result=model.train_predict(xTrain,yTrain,xTest,yTest)
        time=[]
        training_mse = []
        testing_mse = []
        step=int(size/bs)
        for key in range(0,step):
            time.append(result[key]["time"])
            training_mse.append(result[key]["train-mse"])
            testing_mse.append(result[key]["test-mse"])
        train.append(training_mse)
        test.append(testing_mse)
        times.append(time)

    # plot training
    plt.plot(times[0],train[0],color="red",label="Stochastic")
    plt.plot(times[1],train[1],color="blue",label="Mini-batch b=10")
    plt.plot(times[2],train[2],color="yellow",label="Mini-batch b=430")
    plt.plot(times[3],train[3],color="green",label="Mini-batch b=1290")
    plt.plot(times[4],train[4],color="purple",label="Full")
    plt.scatter(time_cost,train_error,label="closed form")
    plt.xlabel("computation time (s)")
    plt.ylabel("Training MSE")
    plt.legend()
    plt.savefig("4atrain.png")
    plt.show()

    # plot testing
    plt.plot(times[0], test[0], color="red", label="Stochastic")
    plt.plot(times[1], test[1], color="blue", label="Mini-batch b=10")
    plt.plot(times[2], test[2], color="yellow", label="Mini-batch b=430")
    plt.plot(times[3], test[3], color="green", label="Mini-batch b=1290")
    plt.plot(times[4], test[4], color="purple", label="Full")
    plt.scatter(time_cost, test_error, label="closed form")
    plt.xlabel("computation time (s)")
    plt.ylabel("Testing MSE")
    plt.legend()
    plt.savefig("4atest.png")
    plt.show()


















if __name__ == "__main__":
    main()