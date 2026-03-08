import numpy as np
from sklearn.cluster import AgglomerativeClustering

def cluster_documents(tfidf_matrix, n_clusters):

    non_zero_indices = np.where(tfidf_matrix.getnnz(axis=1) > 0)[0]
    filtered_matrix = tfidf_matrix[non_zero_indices]

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )

    labels = model.fit_predict(filtered_matrix.toarray())

    return labels, filtered_matrix


def compute_centroids(filtered_matrix, labels, n_clusters):

    centroids = []

    for i in range(n_clusters):
        cluster_docs = filtered_matrix[labels == i]
        centroid = np.asarray(cluster_docs.mean(axis=0))
        centroids.append(centroid)

    return centroids