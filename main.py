from preprocessing import load_cranfield_documents, preprocess_text
from vectorization import vectorize_documents
from clustering import cluster_documents, compute_centroids
from core.retrieval import linear_search, cluster_search
import numpy as np
from core.evaluation import load_queries, load_qrels, precision_at_k

# ------------------------------
# Load Documents
# ------------------------------
docs = load_cranfield_documents("data/cranfield/cran.all.1400")
print("Total documents:", len(docs))

# ------------------------------
# Preprocess
# ------------------------------
clean_docs = [preprocess_text(doc) for doc in docs]

# ------------------------------
# TF-IDF
# ------------------------------
tfidf_matrix, vectorizer = vectorize_documents(clean_docs)
print("TF-IDF Shape:", tfidf_matrix.shape)

# ------------------------------
# Clustering
# ------------------------------
n_clusters = 50
labels, valid_indices = cluster_documents(tfidf_matrix, n_clusters)

print("Clustering completed.")

for i in range(n_clusters):
    print(f"Cluster {i}: {np.sum(labels == i)} documents")

# Compute centroids
filtered_matrix = tfidf_matrix[valid_indices]
centroids = compute_centroids(filtered_matrix, labels, n_clusters)

print("Centroids computed.")

# ------------------------------
# Retrieval Test
# ------------------------------
query = "aerodynamic flow pressure wing theory"

print("\nRunning Linear Search...")
linear_results, linear_time = linear_search(query, vectorizer, tfidf_matrix)
print("Linear Search Time:", linear_time)

print("\nRunning Cluster-Based Search...")
cluster_results, cluster_time = cluster_search(
    query,
    vectorizer,
    filtered_matrix,
    centroids,
    labels
)
print("Cluster Search Time:", cluster_time)

print("\nTop 5 Linear Results:", linear_results[:5])
print("Top 5 Cluster Results:", cluster_results[:5])

# ------------------------------
# Evaluation
# ------------------------------

queries = load_queries("data/cranfield/cran.qry")
qrels = load_qrels("data/cranfield/cranqrel")

print("\nRunning Evaluation on First 20 Queries...")

linear_precisions = []
cluster_precisions = []

for qid in range(20):

    query = preprocess_text(queries[qid])

    # Linear
    linear_results, _ = linear_search(query, vectorizer, tfidf_matrix)
    p_linear = precision_at_k(linear_results, qrels.get(qid, []), k=10)
    linear_precisions.append(p_linear)

    # Cluster
    cluster_results, _ = cluster_search(
        query,
        vectorizer,
        filtered_matrix,
        centroids,
        labels
    )
    p_cluster = precision_at_k(cluster_results, qrels.get(qid, []), k=10)
    cluster_precisions.append(p_cluster)

print("\nAverage Precision@10 (Linear):", sum(linear_precisions)/20)
print("Average Precision@10 (Cluster):", sum(cluster_precisions)/20)