
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.decomposition import PCA

def main():

    xTrain=pd.read_csv("q4xTrain.csv")
    xTest=pd.read_csv("q4xTest.csv")
    yTrain=pd.read_csv("q4yTrain.csv")
    yTest=pd.read_csv("q4yTest.csv")

    ## normalized data
    scaler=StandardScaler()
    xTrain_normal=scaler.fit_transform(xTrain)
    xTest_normal=scaler.transform(xTest)
    model=LogisticRegression()
    model.fit(xTrain_normal,np.ravel(yTrain))
    y_predict=model.predict_proba(xTest_normal)
    fpr,tpr,_=metrics.roc_curve(yTest,y_predict[:,1])
    print("Normalized AUC:",metrics.roc_auc_score(yTest,y_predict[:,1]))
    plt.plot(fpr,tpr,color="red",label="normalized")


    # pca data
    pca=PCA(n_components=5)
    xTrain_pca=pca.fit_transform(xTrain_normal)
    xTest_pca=pca.transform(xTest_normal)
    model=LogisticRegression()
    model.fit(xTrain_pca,np.ravel(yTrain))
    y_pred=model.predict_proba(xTest_pca)
    fpr, tpr, _ = metrics.roc_curve(yTest, y_pred[:, 1])
    print("PCA AUC:",metrics.roc_auc_score(yTest, y_pred[:, 1]))
    plt.plot(fpr, tpr, color="blue", label="PCA")
    plt.ylabel("true positive rate")
    plt.xlabel("false positive rate")
    plt.legend()
    plt.savefig("1c.png")
    plt.show()





if __name__ == "__main__":
    main()