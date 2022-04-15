# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:20:50 2020

@author: Kremena Ivanov
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster import hierarchy
import matplotlib as mpl
from matplotlib.pyplot import cm
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import matplotlib.patches as mpatches


pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

# In[]
'''DATA'''

X = pd.read_csv('bow.csv', index_col=0)
recipes = pd.read_csv('recipes_cleaned.csv', usecols=['id', 'name', 'category_main','category_sub'], index_col=0)

# In[]
'''Hierarchical agglomerative clustering'''

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
fig, ax = plt.subplots(figsize=(10, 7))
plt.title('Hierarchical agglomerative clustering\nwith '+str(model.get_params()['linkage'])+' linkage')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=4)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

fig.savefig('dendrogram.png', dpi=300)

len(model.labels_)
model.n_clusters_
model.n_leaves_
model.n_connected_components_
model.children_.shape


# In[]

'''NEW CODE dendrogram MODEL
https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318
'''


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 8, affinity = 'euclidean', linkage ='ward')
y_hc=hc.fit_predict(X)

model = MDS(n_components=2, random_state=1)
out = model.fit_transform(X)

fig, ax = plt.subplots(figsize=(10,7)) 
ax.scatter(out[y_hc==0, 0], out[y_hc==0, 1], label ='Cluster 1', marker='x', color='silver')
ax.scatter(out[y_hc==1, 0], out[y_hc==1, 1], label ='Cluster 2', marker='.', color='silver')
ax.scatter(out[y_hc==2, 0], out[y_hc==2, 1], label ='Cluster 3', marker='v', color='silver')
ax.scatter(out[y_hc==3, 0], out[y_hc==3, 1], label ='Cluster 4', marker='1', color='silver')
ax.scatter(out[y_hc==4, 0], out[y_hc==4, 1], label ='Cluster 5', marker='*', color='silver')
ax.scatter(out[y_hc==5, 0], out[y_hc==5, 1], label ='Cluster 6', marker='+', color='silver')
ax.scatter(out[y_hc==6, 0], out[y_hc==6, 1], label ='Cluster 7', marker='_', color='silver')
plt.scatter(out[y_hc==7, 0], out[y_hc==7, 1], label ='Cluster 8', marker='|', color='royalblue')
plt.title('Clusters of Recipes (Hierarchical Clustering Model)')
plt.legend(loc='upper left', title='Clusters')
plt.show()

#fig.savefig('cl_8.png', dpi=300)









# In[]
''' MDS & k-means clustering '''
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import matplotlib.patches as mpatches

#MDS
model = MDS(n_components=2, random_state=1)
out = model.fit_transform(X)

#kmeans
model = SpectralClustering(n_clusters=8, affinity='nearest_neighbors', assign_labels='kmeans')
labels = model.fit_predict(out)

plt.style.use('seaborn-notebook')
fig, ax = plt.subplots(figsize=(10,7))
scatter = ax.scatter(out[:, 0], out[:, 1], c=labels, s=50)
ax.title.set_text('Multidimensional scaling with\n k-means clustering - 8 clusters')
ax.title.set_size(16)
legend = ax.legend(*scatter.legend_elements(),loc="lower right", title="Clusters")
ax.add_artist(legend)

plt.show()

fig.savefig('mds_kmeans8.png', dpi=300)


# In[]

'''ANALYSING LABELS'''

table = pd.DataFrame(y_hc, index=X.index, columns=['cluster'])
table.index.name = 'id'

table = pd.merge(left=table, right=recipes, left_index=True, right_index=True)
table = pd.merge(left=table, right=X, left_index=True, right_index=True)

uniqueClusters = sorted(table.cluster.unique())

#create a data frame dictionary to store your data frames
DataFrameDict = {elem : pd.DataFrame for elem in uniqueClusters}

for key in DataFrameDict.keys():
    DataFrameDict[key] = table[:][table.cluster == key]

'''Most popular 10 ingredients'''
c = 7
DataFrameDict[c].shape
DataFrameDict[c].iloc[:,4:].sum().sort_values(ascending=False)[DataFrameDict[c].iloc[:,4:].sum()>10]
DataFrameDict[c].iloc[:,[1,3]].sort_values(by='category_sub',ascending=False)

pd.crosstab(DataFrameDict[c]['category_sub'],columns='freq').sort_values(by='freq', ascending=False)



pd.crosstab(recipes['category_main'],columns='freq').sort_values(by='freq', ascending=False)


    for i in range(8):
       print(DataFrameDict[i].shape)
