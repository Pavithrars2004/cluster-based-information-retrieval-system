import pandas as pd
from core.clustering import cluster_documents, compute_centroids
from core.retrieval import cluster_search
from core.evaluation import precision_at_k
from preprocessing import preprocess_text


def run_experiments(tfidf_matrix, vectorizer, queries, qrels,
                    cluster_options, top_k_options):

    results = []

    for n_clusters in cluster_options:

        labels, filtered_matrix = cluster_documents(tfidf_matrix, n_clusters)
        centroids = compute_centroids(filtered_matrix, labels, n_clusters)

        for top_k in top_k_options:

            precisions = []

            for qid in range(20):

                query = preprocess_text(queries[qid])

                ranked, _ = cluster_search(
                    query,
                    vectorizer,
                    filtered_matrix,
                    centroids,
                    labels,
                    top_k
                )

                p = precision_at_k(ranked, qrels.get(qid, []), 10)
                precisions.append(p)

            avg_precision = sum(precisions) / len(precisions)

            results.append({
                "Clusters": n_clusters,
                "Top_K_Clusters": top_k,
                "Precision@10": avg_precision
            })

    return pd.DataFrame(results)