import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure


## load and organize dataset
iris=datasets.load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
mapping=dict(enumerate(iris['target_names']))
df=df.replace({"target":mapping})
df=df.rename(columns={"target":"species"})


## boxplot
boxplotSepalLen=df.boxplot(column="sepal length (cm)", by="species", grid=False, fontsize=12)
plt.savefig("boxSepalLen.png")
plt.show()

boxplotSepalWid=df.boxplot(column="sepal width (cm)", by="species",grid=False, fontsize=12)
plt.savefig("boxSepalWid.png")
plt.show()

boxplotPetalLen=df.boxplot(column="petal length (cm)", by="species", grid=False, fontsize=12)
plt.savefig("boxPetalLen.png")
plt.show()

boxplotPetalWid=df.boxplot(column="petal width (cm)", by="species", grid=False, fontsize=12)
plt.savefig("boxPetalWid.png")
plt.show()



## scatterplot
sns.scatterplot(x=df["sepal length (cm)"],y=df["sepal width (cm)"],hue="species",data=df)
plt.title("Scatterplot of sepal width and length")
plt.savefig("scatterSepal.png")
plt.show()

ax=sns.scatterplot(x=df["petal length (cm)"],y=df["petal width (cm)"],hue="species",data=df)
plt.title("Scatterplot of petal width and length")
plt.savefig("scatterPetal.png")
plt.show()


