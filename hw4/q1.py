import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import csv


def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """
    data=pd.read_csv(filename)
    labels=[]
    texts=[]
    for index, row in data.iterrows():
        paragraph=row[:][0]
        label=paragraph[0:1]
        text=paragraph[1:]
        labels.append(label)
        texts.append(text)

    X=pd.DataFrame()
    X["text"]=texts
    y=pd.DataFrame()
    y["label"]=labels

    xTrain,xTest,yTrain,yTest=train_test_split(X,y,test_size=0.2)
    return xTrain,xTest,yTrain,yTest


def build_vocab_map(data):
    dict={}
    popular={}
    for index, row in data.iterrows():
        appear={}
        text=row["text"]
        words=text.split(" ")
        for word in words:
            if(appear.get(word)==None):
                appear[word]=True
                if(dict.get(word)==None):
                    dict[word]=1
                else:
                    dict[word]=dict[word]+1
    for key in dict:
        if(dict[key]>=30):
            popular[key]=dict[key]

    return popular


def construct_binary(dictionary,xTrain,xTest,yTrain,yTest):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """

    ## training set
    train_lists=[]
    for index, row in xTrain.iterrows():
        list=[]
        words=row["text"].split(" ")
        for key in dictionary:
            if(key in words):
                list.append(1)
            else:
                list.append(0)
        train_lists.append(list)

    xTrain_binary=np.asarray(train_lists)
    yTrain_binary=np.asarray(yTrain["label"])

    ## testing set
    test_lists=[]
    for index,row in xTest.iterrows():
        list=[]
        words=row["text"].split(" ")
        for key in dictionary:
            if(key in words):
                list.append(1)
            else:
                list.append(0)
        test_lists.append(list)

    xTest_binary=np.asarray(test_lists)
    yTest_binary=np.asarray(yTest["label"])

    return xTrain_binary,xTest_binary,yTrain_binary,yTest_binary


def count(words,target):
    num=0
    for word in words:
        if(word==target):
            num+=1
    return num


def construct_count(dictionary,xTrain,xTest,yTrain,yTest):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """

    ## training set
    train_lists=[]
    for index,row in xTrain.iterrows():
        list=[]
        words=row["text"].split(" ")
        for key in dictionary:
            list.append(count(words,key))
        train_lists.append(list)


    xTrain_count=np.array(train_lists)
    yTrain_count=np.array(yTrain["label"])

    ## testing set
    test_lists=[]
    for index,row in xTest.iterrows():
        list=[]
        words=row["text"].split(" ")
        for key in dictionary:
            list.append(count(words,key))
        test_lists.append(list)

    xTest_count=np.array(test_lists)
    yTest_count=np.array(yTest["label"])

    return xTrain_count,xTest_count,yTrain_count,yTest_count


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    xTrain,xTest,yTrain,yTest=model_assessment(args.data)
    xTrain.to_csv("xTrain.csv",index=False)
    xTest.to_csv("xTest.csv",index=False)
    yTrain.to_csv("yTrain.csv",index=False)
    yTest.to_csv("yTest.csv",index=False)


    # question b
    training=pd.read_csv("xTrain.csv")
    popular_words=build_vocab_map(training)

    # question c
    xTrain=pd.read_csv("xTrain.csv")
    yTrain=pd.read_csv("yTrain.csv")
    xTest=pd.read_csv("xTest.csv")
    yTest=pd.read_csv("yTest.csv")
    xTrain_binary,xTest_binary,yTrain_binary,yTest_binary=construct_binary(popular_words,xTrain,xTest,yTrain,yTest)
    np.savetxt("xTrain_binary.csv",xTrain_binary,delimiter=",")
    np.savetxt("yTrain_binary.csv", yTrain_binary, delimiter=",")
    np.savetxt("xTest_binary.csv", xTest_binary, delimiter=",")
    np.savetxt("yTest_binary.csv",yTest_binary,delimiter=",")


    # question d
    xTrain = pd.read_csv("xTrain.csv")
    yTrain = pd.read_csv("yTrain.csv")
    xTest = pd.read_csv("xTest.csv")
    yTest = pd.read_csv("yTest.csv")
    xTrain_count,xTest_count,yTrain_count,yTest_count=construct_count(popular_words,xTrain,xTest,yTrain,yTest)
    np.savetxt("xTrain_count.csv",xTrain_count,delimiter=",")
    np.savetxt("yTrain_count.csv",yTrain_count,delimiter=",")
    np.savetxt("xTest_count.csv",xTest_count,delimiter=",")
    np.savetxt("yTest_count.csv",yTest_count,delimiter=",")





if __name__ == "__main__":
    main()
