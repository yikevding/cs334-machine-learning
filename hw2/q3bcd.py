import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def main():
    ## load datasets
    xTrain = pd.read_csv("q4xTrain.csv").to_numpy()
    yTrain = pd.read_csv("q4yTrain.csv").to_numpy().ravel()
    xTest = pd.read_csv("q4xTest.csv").to_numpy()
    yTest = pd.read_csv("q4yTest.csv").to_numpy().ravel()

    ## question 3b
    print("knn----------")
    optimal_k = 15  # acquired from doing question a

    # train on entire datasets
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(xTrain, yTrain)
    yPredict = knn.predict(xTest)
    accuracy1 = accuracy_score(yTest, yPredict)
    print("Training on entire train datasets")
    print("Accuracy: ", accuracy1)

    yPredict = knn.predict_proba(xTest)
    auc1 = roc_auc_score(y_true=yTest, y_score=yPredict[:, 1])
    print("AUC: ", auc1)

    # remove 5% of the original training datasets
    xTrain2, xValid, yTrain2, yValid = train_test_split(xTrain, yTrain, test_size=0.05)
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(xTrain2, yTrain2)
    yPredict = knn.predict(xTest)
    accuracy2 = accuracy_score(y_true=yTest, y_pred=yPredict)
    print("Training on 95% train datasets")
    print("Accuracy: ", accuracy2)
    yPredict = knn.predict_proba(xTest)
    auc2 = roc_auc_score(y_true=yTest, y_score=yPredict[:, 1])
    print("AUC: ", auc2)

    # remove 10% of the original training datasets
    xTrain3, xValid, yTrain3, yValid = train_test_split(xTrain, yTrain, test_size=0.1)
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(xTrain3, yTrain3)
    yPredict = knn.predict(xTest)
    accuracy3 = accuracy_score(y_true=yTest, y_pred=yPredict)
    print("Training on 90% train datasets")
    print("Accuracy: ", accuracy3)
    yPredict = knn.predict_proba(xTest)
    auc3 = roc_auc_score(y_true=yTest, y_score=yPredict[:, 1])
    print("AUC: ", auc3)

    # remove 20% of the original training datasets
    xTrain4, xValid, yTrain4, yValid = train_test_split(xTrain, yTrain, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(xTrain4, yTrain4)
    yPredict = knn.predict(xTest)
    accuracy4 = accuracy_score(y_true=yTest, y_pred=yPredict)
    print("Training on 80% train datasets")
    print("Accuracy: ", accuracy4)
    yPredict = knn.predict_proba(xTest)
    auc4 = roc_auc_score(y_true=yTest, y_score=yPredict[:, 1])
    print("AUC: ", auc4)



    ## question 3c
    print("decision tree----------")
    ## data acquired from question a
    optimal_criterion = "gini"
    optimal_depth = 5
    optimal_min_leaf = 51

    ## train on the entire training datasets
    tree = DecisionTreeClassifier(criterion=optimal_criterion, max_depth=optimal_depth,
                                  min_samples_leaf=optimal_min_leaf)
    tree.fit(xTrain, yTrain)
    yPredict = tree.predict(xTest)
    accuracy5 = accuracy_score(yTest, yPredict)
    print("Training on entire train datasets")
    print("Accuracy: ", accuracy5)

    yPredict = tree.predict_proba(xTest)
    auc5 = roc_auc_score(y_true=yTest, y_score=yPredict[:, 1])
    print("AUC: ", auc5)

    # remove 5% of the original training datasets
    xTrain6, xValid, yTrain6, yValid = train_test_split(xTrain, yTrain, test_size=0.05)
    tree = DecisionTreeClassifier(criterion=optimal_criterion, max_depth=optimal_depth,
                                  min_samples_leaf=optimal_min_leaf)
    tree.fit(xTrain6, yTrain6)
    yPredict = tree.predict(xTest)
    accuracy6 = accuracy_score(yTest, yPredict)
    print("Training on 95% train datasets")
    print("Accuracy: ", accuracy6)

    yPredict = tree.predict_proba(xTest)
    auc6 = roc_auc_score(y_true=yTest, y_score=yPredict[:, 1])
    print("AUC: ", auc6)

    # remove 10% of the original training datasets
    xTrain7, xValid, yTrain7, yValid = train_test_split(xTrain, yTrain, test_size=0.1)
    tree = DecisionTreeClassifier(criterion=optimal_criterion, max_depth=optimal_depth,
                                  min_samples_leaf=optimal_min_leaf)
    tree.fit(xTrain7, yTrain7)
    yPredict = tree.predict(xTest)
    accuracy7 = accuracy_score(yTest, yPredict)
    print("Training on 90% train datasets")
    print("Accuracy: ", accuracy7)

    yPredict = tree.predict_proba(xTest)
    auc7 = roc_auc_score(y_true=yTest, y_score=yPredict[:, 1])
    print("AUC: ", auc7)

    # remove 20% of the original training datasets
    xTrain8, xValid, yTrain8, yValid = train_test_split(xTrain, yTrain, test_size=0.2)
    tree = DecisionTreeClassifier(criterion=optimal_criterion, max_depth=optimal_depth,
                                  min_samples_leaf=optimal_min_leaf)
    tree.fit(xTrain8, yTrain8)
    yPredict = tree.predict(xTest)
    accuracy8 = accuracy_score(yTest, yPredict)
    print("Training on 80% train datasets")
    print("Accuracy: ", accuracy8)

    yPredict = tree.predict_proba(xTest)
    auc8 = roc_auc_score(y_true=yTest, y_score=yPredict[:, 1])
    print("AUC: ", auc8)

    table=pd.DataFrame([
        ["KNN original",accuracy1,auc1],
        ["KNN 95%",accuracy2,auc2],
        ["KNN 90%",accuracy3,auc3],
        ["KNN 80%",accuracy4,auc4],
        ["Decision tree original",accuracy5,auc5],
        ["Decision tree 95%",accuracy6,auc6],
        ["Decison tree 90%",accuracy7,auc7],
        ["Decision tree 80%",accuracy8,auc8]],
        columns=["Strategy","Accuracy","AUC"])
    print(table)

if __name__ == "__main__":
    main()

