#Using k-means for customer segmentation

import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
import pandas as pd


cust_df = pd.read_csv("Cust_Segmentation.csv")
print(cust_df.head())

'''As you can see, __Address__ in this dataset is a categorical variable.
k-means algorithm isn't directly applicable to categorical variables because
Euclidean distance function isn't really meaningful for discrete variables.
So, lets drop this feature and run clustering.'''


df = cust_df.drop('Address', axis=1)
print(df.head())


'''Now let's normalize the dataset. But why do we need normalization in the first place?
Normalization is a statistical method that helps mathematical-based algorithms to interpret
features with different magnitudes and distributions equally. We use StandardScaler() to
normalize our dataset.
'''
from sklearn.preprocessing import StandardScaler #Normalization
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)


clusterNum = 3 #sklearn Learn Library 
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


df["Clus_km"] = labels
print(df.head(5))
df.groupby('Clus_km').mean()



area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
#plt.ylabel('Age', fontsize=18)
#plt.xlabel('Income', fontsize=16)
#plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))

plt.show()
