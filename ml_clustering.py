import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing







# split data - training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X_tips, y_tips, random_state=1, test_size = 0.3) # dataset split: 70% training and 30% test

#normalized data
Xtrain_n = preprocessing.normalize(Xtrain)
Xtest_n = preprocessing.normalize(Xtest)

#standardized data
Xtrain_s = preprocessing.scale(Xtrain)
Xtest_s = preprocessing.scale(Xtest)


#random forest
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


print('Random forest models')
print('Grid search with five-fold cross-validation')

param_grid = {'max_depth':np.arange(1,11),
             'random_state': [0]}

grid = GridSearchCV(RandomForestClassifier(),param_grid, cv=5, return_train_score=True)

#original data
print('\nOriginal data:')
grid.fit(Xtrain, ytrain.values.ravel())
score_rf_od = grid.score(Xtest,ytest.values.ravel())
print(f'Best mean cross-validation accuracy: {grid.best_score_:.3f}')
print(f'Grid search best parameters: {grid.best_params_}')
print(f'Test accuracy: {score_rf_od:.3f}')



#normalization & standardization

from sklearn import preprocessing

#normalized data
X_n = preprocessing.normalize(X)
X_n = pd.DataFrame(data = X_n, columns=X.columns, index = X.index)

#standardized data
X_s = preprocessing.scale(X)
X_s = pd.DataFrame(data = X_s, columns=X.columns, index = X.index)


#plot a dendrogram for one of the hierarchical clustering methods
import scipy.cluster.hierarchy as sch

print('Hierarchical agglomerative clustering\nFull dendrogram\n')

plt.figure(figsize=(10, 7))
plt.title('Original data, ward linkage')
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()

plt.figure(figsize=(10, 7))
plt.title('Normalzed data, ward linkage')
dendrogram = sch.dendrogram(sch.linkage(X_n, method='ward'))
plt.show()

plt.figure(figsize=(10, 7))
plt.title('Standardized data, ward linkage')
dendrogram = sch.dendrogram(sch.linkage(X_s, method='ward'))
plt.show()


import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


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

print('Hierarchical agglomerative clustering\Concise dendrogram\n')

model = model.fit(X)
plt.figure(figsize=(10, 7))
plt.title('Original data, '+str(model.get_params()['linkage'])+' linkage')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

model = model.fit(X_n)
plt.figure(figsize=(10, 7))
plt.title('Normalized data, '+str(model.get_params()['linkage'])+' linkage')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

model = model.fit(X_s)
plt.figure(figsize=(10, 7))
plt.title('Standardized data, '+str(model.get_params()['linkage'])+' linkage')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# MDS & k-means clustering

from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

#MDS
model = MDS(n_components=2, random_state=1)
out = model.fit_transform(X) #original data
out_n = model.fit_transform(X_n) #normalized data
out_s = model.fit_transform(X_s) #standardized data

#kmeans
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
labels = model.fit_predict(out) #original data
labels_n = model.fit_predict(out_n) #normalized data
labels_s = model.fit_predict(out_s) #standardized data


#original data
fig = plt.figure(figsize=(20,7))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

scatter1 = ax1.scatter(out[:,0], out[:,1], c=X['cylinders'], cmap='plasma')
ax1.title.set_text('Original data - Colors by number of cylinders')
ax1.title.set_size(16)
legend1 = ax1.legend(*scatter1.legend_elements(),loc="lower right", title="Cylinders")
ax1.add_artist(legend1)

scatter2 = ax2.scatter(out[:, 0], out[:, 1], c=labels, s=50, cmap='viridis')
ax2.title.set_text('Original data - Two clusters')
ax2.title.set_size(16)
legend2 = ax2.legend(*scatter2.legend_elements(),loc="lower right", title="Clusters")
ax2.add_artist(legend2)

plt.show()
