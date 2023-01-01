# PCA and Clustering (Kmeans) examples

In the following post we will analyze a coolab notebook containing two examples, one in which we will perform a principal component analysis (PCA), with the help of
the iris dataset. And in the other example we will do clustering with kmeans analyzing the data from the diabetes dataset.

## PCA

We know that when facing a ML problem, we may have datasets with many characteristics which may represent a dimensionality problem, adding problems such as:
*  high probability of overfitting.
*  visualization difficulties
*  affects the speed of the training process.

In this case we will do the PCA to the iris dataset, analyzing the dataset with different characteristics, all this is possible thanks to the Singular Value Decomposition (SVD) technique, which is a standard matrix factorization technique.In code this technique was implemented in the following way:

```python
#The Singular Value Decomposition (SVD)
x_centered = np.empty((150,4))
x_centered = x - x.mean(axis=0)

U, s, Vt = np.linalg.svd(x_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]
c3 = Vt.T[:, 2]

W2 = Vt.T[:, :3]
X2D = x_centered.dot(W2)

```
Where we obtain the following distributions of our data according to the characteristics.

**PC1**


![image](https://user-images.githubusercontent.com/115313115/210160167-33382209-f75c-4f1b-b25e-a69056cfdb4d.png)

**PC2**


![image](https://user-images.githubusercontent.com/115313115/210160177-f8df8b3c-0029-4d14-a80c-b8a27f808aa6.png)

**PC3**


![image](https://user-images.githubusercontent.com/115313115/210160181-41236d9b-c797-4cae-bbcb-eb9c5c882ee3.png)

**Dimension vs variance**


![image](https://user-images.githubusercontent.com/115313115/210160187-690a4351-e226-49c2-8406-db79268ae9ab.png)

We can also perform the analysis with the sklearn library as follows.
```python
#Using sklearn
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
pca.fit(x)
X2D_sklearn = pca.transform(x)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection = '3d')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.scatter(X2D_sklearn[:,0], X2D_sklearn[:,1], X2D_sklearn[:,2], c=y, cmap=my_cmap)

plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.grid()
plt.show()

```

## Kmeans example
In the following example we will analyze a clustering problem, using Kmeans and the 'Pima-Indians-Diabetes-Dataset'. Where we will first normalize our dataset, in order to assemble our kmeans model with a random number of clusters, once the analysis is done with their respective predictions we will proceed to interpret the elbow graph to see which is the right number of clusters for our exercise. Some code excerpts relevant to the process:
```python
#Create model
from sklearn.cluster import KMeans
k = 2
km = KMeans(n_clusters=k,init='random',)
km.fit(x_train)

#Predict
test_data = x_test[2,:]
test_data =test_data.reshape((1,8))
print(test_data)
predict = km.predict(test_data)
print(predict)

```
**Elbow graph**


![image](https://user-images.githubusercontent.com/115313115/210160504-82d162fa-cbf4-4352-9dc8-9e359f8430e3.png)

Once the number of clusters has been determined, we will proceed to do a silhouette analysis to interpret a little of how our clusters work using a Kmeans ++ model.

```python
k = 2
km = KMeans(n_clusters=k,init='k-means++')
km.fit(x_train)

from sklearn.metrics import silhouette_samples
from matplotlib import cm
y_km = km.predict(x_train)
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]

silhouette_vals = silhouette_samples(x_train, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []

for i, c in enumerate(cluster_labels):
  c_silhouette_vals = silhouette_vals[y_km == c]
  c_silhouette_vals.sort()
  y_ax_upper += len(c_silhouette_vals)
  color = cm.jet(float(i) / n_clusters)
  plt.barh(range(y_ax_lower, y_ax_upper),
           c_silhouette_vals,
           height=1.0,
           edgecolor='none',
           color=color)
  yticks.append((y_ax_lower + y_ax_upper) / 2.)
  y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
              color="red",
              linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()

```
**Silhoette analisis**


![image](https://user-images.githubusercontent.com/115313115/210160601-c4221848-1f63-406f-9240-83ba9ed3a08f.png)
