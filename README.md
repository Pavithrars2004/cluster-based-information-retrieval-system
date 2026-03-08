# Hierarchical Cluster-Based Information Retrieval System

A scalable **Information Retrieval System (IRS)** that compares **traditional linear search** with **hierarchical cluster-based retrieval** using the **Cranfield Aeronautics Collection**.  
The project demonstrates how clustering can reduce search space and improve retrieval efficiency while maintaining acceptable precision.

---

## Project Overview

Information Retrieval Systems are designed to retrieve relevant documents from large collections of text data.  
Traditional retrieval methods perform **linear search**, where a user query is compared with every document in the dataset. While this ensures completeness, it becomes inefficient as the dataset grows.

This project implements a **cluster-based retrieval strategy** where documents are grouped using **hierarchical clustering**.  
Instead of searching the entire collection, the query is first matched with cluster representatives, and only the most relevant clusters are searched.

This approach significantly reduces the number of similarity computations and improves scalability.

---

## Dataset

**Cranfield Aeronautics Collection**

- 1400 aerospace research documents
- Technical reports related to aerodynamics and fluid mechanics
- Common benchmark dataset used in Information Retrieval research

---

## System Architecture

The system consists of two main stages.

### Offline Indexing Phase
1. Document Collection
2. Text Preprocessing
3. TF-IDF Vectorization
4. Hierarchical Clustering
5. Cluster Index Creation

### Online Query Processing Phase
1. User Query Input
2. Query Preprocessing
3. Query Vectorization
4. Cluster Selection
5. Cosine Similarity Computation
6. Ranked Document Retrieval

---

## Technologies Used

- Python
- Scikit-learn
- NLTK
- NumPy
- Pandas
- SciPy
- Matplotlib
- Streamlit (for interactive demo)

---

## Key Techniques Implemented

### Text Preprocessing
- Tokenization
- Stopword removal
- Lowercasing

### Vector Representation
- TF-IDF (Term Frequency–Inverse Document Frequency)

### Similarity Measurement
- Cosine Similarity

### Clustering
- Hierarchical Agglomerative Clustering

### Evaluation Metrics
- Precision@10
- Recall@10
- Query Processing Time

---

## Experimental Results

The system compares **Linear Search vs Cluster-Based Retrieval**.

Observations:

- Linear search provides slightly higher precision because it examines every document.
- Cluster-based retrieval significantly reduces search space.
- Selecting top clusters improves efficiency while maintaining acceptable accuracy.

This demonstrates the **efficiency–effectiveness trade-off** in Information Retrieval.

---
