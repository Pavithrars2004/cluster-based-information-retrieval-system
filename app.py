import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import time

from preprocessing import load_cranfield_documents, preprocess_text
from vectorization import vectorize_documents
from core.clustering import cluster_documents, compute_centroids
from core.retrieval import linear_search, cluster_search
from core.evaluation import (
    load_queries,
    load_qrels,
    precision_at_k,
    recall_at_k
)
from core.experiments import run_experiments
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(layout="wide")

st.title("Hierarchical Cluster-Based Information Retrieval System")
st.markdown("### Cranfield Collection Experimental Analysis")

# Load Data
docs = load_cranfield_documents("data/cranfield/cran.all.1400")
clean_docs = [preprocess_text(d) for d in docs]
tfidf_matrix, vectorizer = vectorize_documents(clean_docs)

queries = load_queries("data/cranfield/cran.qry")
qrels = load_qrels("data/cranfield/cranqrel")

# Sidebar Settings
st.sidebar.header("Experiment Settings")

cluster_options = st.sidebar.multiselect(
    "Select Number of Clusters",
    [20, 50],
    default=[20, 50]
)

top_k_options = st.sidebar.multiselect(
    "Select Top-K Clusters",
    [1, 2, 3],
    default=[1, 2]
)

# Run Experiments
if st.sidebar.button("Run Experiments"):

    linear_precisions = []
    linear_recalls = []
    linear_times = []

    for qid in range(20):
        query = preprocess_text(queries[qid])

        start = time.time()
        ranked, _ = linear_search(query, vectorizer, tfidf_matrix)
        linear_times.append(time.time() - start)

        linear_precisions.append(
            precision_at_k(ranked, qrels.get(qid, []), 10)
        )

        linear_recalls.append(
            recall_at_k(ranked, qrels.get(qid, []), 10)
        )

    linear_avg_precision = sum(linear_precisions) / len(linear_precisions)
    linear_avg_recall = sum(linear_recalls) / len(linear_recalls)
    linear_avg_time = sum(linear_times) / len(linear_times)

    # Cluster experiments
    results = []

    for n_clusters in cluster_options:

        labels, filtered_matrix = cluster_documents(tfidf_matrix, n_clusters)
        centroids = compute_centroids(filtered_matrix, labels, n_clusters)

        for top_k in top_k_options:

            precisions = []
            recalls = []
            times = []

            for qid in range(20):

                query = preprocess_text(queries[qid])

                start = time.time()
                ranked, _ = cluster_search(
                    query,
                    vectorizer,
                    filtered_matrix,
                    centroids,
                    labels,
                    top_k
                )
                times.append(time.time() - start)

                precisions.append(
                    precision_at_k(ranked, qrels.get(qid, []), 10)
                )

                recalls.append(
                    recall_at_k(ranked, qrels.get(qid, []), 10)
                )

            results.append({
                "Method": f"{n_clusters} clusters (Top-{top_k})",
                "Precision@10": sum(precisions)/len(precisions),
                "Recall@10": sum(recalls)/len(recalls),
                "Avg Time (s)": sum(times)/len(times)
            })

    # Add Linear Baseline
    results.insert(0, {
        "Method": "Linear",
        "Precision@10": linear_avg_precision,
        "Recall@10": linear_avg_recall,
        "Avg Time (s)": linear_avg_time
    })

    df = pd.DataFrame(results)

    st.subheader("Results Table")
    st.dataframe(df, use_container_width=True)

    # Graph
    st.subheader("Precision Comparison")

    fig, ax = plt.subplots()
    ax.bar(df["Method"], df["Precision@10"])
    ax.set_ylabel("Precision@10")
    ax.set_xticklabels(df["Method"], rotation=45)
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("Observations")
    st.markdown("""
    - Linear search achieves highest precision due to exhaustive search.
    - Cluster-based retrieval reduces search time.
    - Increasing Top-K improves recall.
    - Trade-off exists between efficiency and effectiveness.
    """)

# -------------------------------
# Live Query Demo
# -------------------------------
st.subheader("Live Query Demo")

user_query = st.text_input("Enter a query:")

if st.button("Search") and user_query:

    query = preprocess_text(user_query)

    ranked_linear, _ = linear_search(query, vectorizer, tfidf_matrix)

    labels, filtered_matrix = cluster_documents(tfidf_matrix, 50)
    centroids = compute_centroids(filtered_matrix, labels, 50)

    ranked_cluster, _ = cluster_search(
        query,
        vectorizer,
        filtered_matrix,
        centroids,
        labels,
        top_k_clusters=2
    )

    st.markdown("### Top 5 Linear Results")
    for idx in ranked_linear[:5]:
        st.write(docs[idx][:300] + "...")

    st.markdown("### Top 5 Cluster Results")
    for idx in ranked_cluster[:5]:
        st.write(docs[idx][:300] + "...")

# -------------------------------
# Dendrogram
# -------------------------------
st.subheader("Dendrogram (First 100 Documents)")

if st.button("Show Dendrogram"):

    Z = linkage(tfidf_matrix.toarray()[:100],
                method='average',
                metric='cosine')

    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")

    st.pyplot(fig)