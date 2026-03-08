from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_documents(documents):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    return tfidf_matrix, vectorizer