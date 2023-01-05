
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

def main():
    # normalize the dataset
    xTrain=pd.read_csv("q4xTrain.csv")
    xTest=pd.read_csv("q4xTest.csv")
    yTrain=pd.read_csv("q4yTrain.csv")
    yTest=pd.read_csv("q4yTest.csv")
    normalizer=StandardScaler()
    normalizer.fit(xTrain)
    train=normalizer.transform(xTrain)
    test=normalizer.transform(xTest)
    xTrain_normal=pd.DataFrame()
    xTest_normal=pd.DataFrame()
    index=0
    for row in xTest:
        xTrain_normal[row]=train[:,index]
        xTest_normal[row]=test[:,index]
        index+=1
    xTrain_normal.to_csv("xTrain_normal.csv",index=False)
    xTest_normal.to_csv("xTest_normal.csv",index=False)

    # train and test logistic model
    model=LogisticRegression(max_iter=1000)
    model.fit(xTrain_normal,yTrain)
    y_pred=model.predict_proba(xTest_normal)
    np.savetxt("test_prob.csv",y_pred,delimiter=",")
    print(y_pred)



if __name__ == "__main__":
    main()