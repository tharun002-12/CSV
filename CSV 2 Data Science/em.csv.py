from numpy import unique, where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
n_clusters_per_class=1, random_state=4)

kmeans_model = KMeans(n_clusters=2)

kmeans_model.fit(X)

yhat_kmeans = kmeans_model.predict(X)

clusters_kmeans = unique(yhat_kmeans)

for cluster in clusters_kmeans:
    
    row_indexes = where(yhat_kmeans == cluster)
    
    pyplot.scatter(X[row_indexes, 0], X[row_indexes, 1])

pyplot.title("K-means Clustering")
pyplot.show()

gmm_model = GaussianMixture(n_components=2)

gmm_model.fit(X)

yhat_gmm = gmm_model.predict(X)

clusters_gmm = unique(yhat_gmm)

for cluster in clusters_gmm:
    
    row_indexes = where(yhat_gmm == cluster)
    
    pyplot.scatter(X[row_indexes, 0], X[row_indexes, 1])

pyplot.title("Gaussian Mixture Clustering")
pyplot.show()
