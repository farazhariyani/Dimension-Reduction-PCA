#import packages
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

#load dataset
wine = pd.read_csv("wine.csv")

wine.describe()

#Normalization
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min()) #min max scaler
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(wine)
df_norm.describe()

#Hierarchical Clustering
z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(2, 14));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 4 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_
h_complete
cluster_labels = pd.Series(h_complete.labels_)
cluster_labels
wine['clust'] = cluster_labels # creating a new column and assigning it to new column 

wine = wine.iloc[:, [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine.head()

# Aggregate mean of each cluster
wine.groupby(wine.clust).mean()

# creating a csv file 
wine.to_csv("PCA-HierarchicalClustering-Assignment.csv", encoding = "utf-8")

import os
os.getcwd()

# K-means clustering
# scree plot or elbow curve 
TWSS = []
k = list(range(2, 14))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
wine['clust'] = mb # creating a  new column and assigning it to new column 

wine.head()
df_norm.head()

wine = wine.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine.head()

wine.iloc[:, 1:14].groupby(wine.clust).mean()

wine.to_csv("PCA-Kmeans.csv", encoding = "utf-8")

#PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 

# Normalizing the numerical data 
uni_normal = scale(wine)
uni_normal

pca = PCA(n_components = 6)
pca_values = pca.fit_transform(uni_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

pca.components_
pca.components_[0]

# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5"
final = pd.concat([wine.Univ, pca_data.iloc[:, 0:3]], axis = 1)

# Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = pca_data.pc0, y = pca_data.pc1)