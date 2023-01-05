import numpy as np
import seaborn as sns
import pandas as pd
import selFeat
import matplotlib.pyplot as plt


def main():

    ## load datasets
    xTrain=pd.read_csv("eng_xTrain.csv")
    yTrain=pd.read_csv("eng_yTrain.csv")
    xTrain=selFeat.extract_features(xTrain)
    xTrain["label"] = yTrain["label"]
    plt.figure(figsize=(15,10))
    corr_matrix=xTrain.corr(method="pearson").round(2)
    heatmap=sns.heatmap(corr_matrix,vmax=1, vmin=-1,annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    plt.savefig("heatmap.png")
    plt.show()















if __name__ == "__main__":
    main()