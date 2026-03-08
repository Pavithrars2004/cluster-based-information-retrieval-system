import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

def linear_search(query, vectorizer, tfidf_matrix):

    start = time.time()

    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]

    ranked = np.argsort(similarities)[::-1]

    return ranked, time.time() - start


def cluster_search(query, vectorizer, filtered_matrix, centroids, labels, top_k_clusters):

    start = time.time()

    query_vec = vectorizer.transform([query])

    centroid_matrix = np.vstack(centroids)
    cluster_sim = cosine_similarity(query_vec, centroid_matrix)[0]

    top_clusters = np.argsort(cluster_sim)[::-1][:top_k_clusters]

    indices = []

    for cid in top_clusters:
        indices.extend(np.where(labels == cid)[0])

    indices = np.array(indices)

    cluster_docs = filtered_matrix[indices]
    similarities = cosine_similarity(query_vec, cluster_docs)[0]

    ranked_local = np.argsort(similarities)[::-1]
    ranked = indices[ranked_local]

    return ranked, time.time() - start