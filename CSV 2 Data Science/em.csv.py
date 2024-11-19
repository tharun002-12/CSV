from numpy import unique, where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
n_clusters_per_class=1, random_state=4)
# K-means clustering
# define the model
kmeans_model = KMeans(n_clusters=2)
# fit the model
kmeans_model.fit(X)
# assign a cluster to each example
yhat_kmeans = kmeans_model.predict(X)
# retrieve unique clusters
clusters_kmeans = unique(yhat_kmeans)
# create scatter plot for samples from each cluster
for cluster in clusters_kmeans:
    # get row indexes for samples with this cluster
    row_indexes = where(yhat_kmeans == cluster)
    # create scatter plot of these samples
    pyplot.scatter(X[row_indexes, 0], X[row_indexes, 1])
# show the plot
pyplot.title("K-means Clustering")
pyplot.show()
# Gaussian Mixture clustering
# define the model
gmm_model = GaussianMixture(n_components=2)
# fit the model
gmm_model.fit(X)
# assign a cluster to each example
yhat_gmm = gmm_model.predict(X)
# retrieve unique clusters
clusters_gmm = unique(yhat_gmm)
# create scatter plot for samples from each cluster
for cluster in clusters_gmm:
    # get row indexes for samples with this cluster
    row_indexes = where(yhat_gmm == cluster)
    # create scatter plot of these samples
    pyplot.scatter(X[row_indexes, 0], X[row_indexes, 1])
# show the plot
pyplot.title("Gaussian Mixture Clustering")
pyplot.show()
