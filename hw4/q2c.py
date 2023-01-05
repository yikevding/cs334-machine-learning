
import pandas as pd
import numpy as np
from q1 import build_vocab_map
from perceptron import Perceptron



def main():

    # load datasets
    xTrain = pd.read_csv("xTrain.csv")
    yTrain = pd.read_csv("yTrain.csv")
    xTest = pd.read_csv("xTest.csv")
    yTest = pd.read_csv("yTest.csv")

    xTrain_binary = np.loadtxt("xTrain_binary.csv", delimiter=",")
    xTest_binary = np.loadtxt("xTest_binary.csv", delimiter=",")
    yTrain_binary = np.loadtxt("yTrain_binary.csv", delimiter=",")
    yTest_binary = np.loadtxt("yTest_binary.csv", delimiter=",")

    xTrain_count = np.loadtxt("xTrain_count.csv", delimiter=",")
    xTest_count = np.loadtxt("xTest_count.csv", delimiter=",")
    yTrain_count = np.loadtxt("yTrain_count.csv", delimiter=",")
    yTest_count = np.loadtxt("yTest_count.csv", delimiter=",")

    popular_words=build_vocab_map(xTrain)
    binary_epoch=11
    count_epoch=33

    # binary set
    model=Perceptron(binary_epoch)
    stats=model.train(xTrain_binary,yTrain_binary)
    weights=model.w
    results={}
    index=1 # index=0 is the weights of the intercept, or bias
    for key in popular_words:
        weight=weights[index]
        index+=1
        results[key]=weight

    # sort dict by values
    ordered=sorted(results.items(),key=lambda word:word[1])
    negative_words=ordered[0:15]
    positive_words=ordered[-15:]
    print("-----Binary dataset-----")
    negative=[]
    for item in negative_words:
        negative.append(item[0])
    positive=[]
    for item in positive_words:
        positive.append(item[0])
    print("15 words with most negative weights: ",negative)
    print("15 words with most positive weights: ",positive)


    # count set
    model=Perceptron(count_epoch)
    stats=model.train(xTrain_count,yTrain_count)
    weights=model.w
    results={}
    index=1
    for key in popular_words:
        weight=weights[index]
        index+=1
        results[key]=weight
    ordered = sorted(results.items(), key=lambda word: word[1])
    negative_words = ordered[0:15]
    positive_words = ordered[-15:]
    print("-----Count dataset-----")
    negative = []
    for item in negative_words:
        negative.append(item[0])
    positive = []
    for item in positive_words:
        positive.append(item[0])
    print("15 words with most negative weights: ",negative)
    print("15 words with most positive weights: ",positive)



if __name__ == "__main__":
    main()