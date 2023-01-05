import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def main():

    ## loading data
    xTrain = pd.read_csv("q4xTrain.csv").to_numpy()
    yTrain = pd.read_csv("q4yTrain.csv").to_numpy().ravel()


    ## find optimal hyperparameters for knn model
    max_k=50
    k_values=list(range(1,max_k+1))
    parameters=dict(n_neighbors=k_values)
    knn=KNeighborsClassifier()
    k_fold=10
    grid=GridSearchCV(estimator=knn,param_grid=parameters,cv=k_fold,scoring="roc_auc")
    grid.fit(xTrain,yTrain)
    print("The best parameter for knn model is ",grid.best_params_)


    ## find optimal hyperparameters for decision tree
    k_fold=10
    max_depth=30
    depth_range=list(range(1,max_depth+1))
    mls=100
    leaf_range=list(range(1,mls+1))
    parameters={"criterion":["gini","entropy"],'max_depth':depth_range,'min_samples_leaf':leaf_range}
    tree=DecisionTreeClassifier(random_state=1) # to control we get the same best parameters every time
    grid=GridSearchCV(estimator=tree,param_grid=parameters,cv=k_fold,scoring="roc_auc")
    grid.fit(xTrain,yTrain)
    print("The best parameters for decision tree model are ", grid.best_params_)

























if __name__ == "__main__":
    main()