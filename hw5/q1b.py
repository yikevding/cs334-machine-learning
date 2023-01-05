

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

def main():
    xTrain=pd.read_csv("xTrain_normal.csv")
    xTest=pd.read_csv("xTest_normal.csv")
    pca=PCA()
    X_pca_train=pca.fit_transform(xTrain)
    X_pca_test=pca.transform(xTest)
    np.savetxt("pca_train.csv",X_pca_train,delimiter=",")
    np.savetxt("pca_test.csv",X_pca_test,delimiter=",")


    # find out how many components need to capture at least 0.95
    count=0
    sum=0
    for variance in pca.explained_variance_ratio_:
        sum+=variance
        count+=1
        if(sum>=0.95):
            break
    print(count,"components needed to capture at least 95% variance of the original data.")

    # check the first 3 components
    pca=PCA(n_components=3)
    X_pca=pca.fit_transform(xTrain)
    print("-----First 3 principal components-----")
    print(pca.components_)








if __name__ == "__main__":
    main()