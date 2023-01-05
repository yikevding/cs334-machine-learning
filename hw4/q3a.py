
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from perceptron import calc_mistakes

def main():
    # load datasets
    xTrain_binary = np.loadtxt("xTrain_binary.csv", delimiter=",")
    xTest_binary = np.loadtxt("xTest_binary.csv", delimiter=",")
    yTrain_binary = np.loadtxt("yTrain_binary.csv", delimiter=",")
    yTest_binary = np.loadtxt("yTest_binary.csv", delimiter=",")

    xTrain_count = np.loadtxt("xTrain_count.csv", delimiter=",")
    xTest_count = np.loadtxt("xTest_count.csv", delimiter=",")
    yTrain_count = np.loadtxt("yTrain_count.csv", delimiter=",")
    yTest_count = np.loadtxt("yTest_count.csv", delimiter=",")

    # train and predict on binary set
    model=MultinomialNB()
    model.fit(xTrain_binary,yTrain_binary)
    train_predicted=model.predict(xTrain_binary)
    train_mistakes=calc_mistakes(train_predicted,yTrain_binary)
    test_predicted=model.predict(xTest_binary)
    test_mistakes=calc_mistakes(test_predicted,yTest_binary)
    print("-----Binary set-----")
    print("Number of mistakes on the training set is ",train_mistakes)
    print("Number of mistakes on the testing set is ",test_mistakes)

    # train and predict on count set
    model = MultinomialNB()
    model.fit(xTrain_count,yTrain_count)
    train_predicted = model.predict(xTrain_count)
    train_mistakes = calc_mistakes(train_predicted, yTrain_count)
    test_predicted = model.predict(xTest_count)
    test_mistakes = calc_mistakes(test_predicted, yTest_count)
    print("-----Count set-----")
    print("Number of mistakes on the training set is ", train_mistakes)
    print("Number of mistakes on the testing set is ", test_mistakes)






if __name__ == "__main__":
    main()