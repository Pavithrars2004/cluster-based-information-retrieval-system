def load_queries(filepath):

    queries = []

    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    blocks = content.split(".I ")[1:]

    for block in blocks:
        parts = block.split(".W")
        if len(parts) > 1:
            queries.append(parts[1].strip())

    return queries


def load_qrels(filepath):

    relevance = {}

    with open(filepath, 'r') as file:
        for line in file:
            qid, did = line.split()[:2]
            qid = int(qid) - 1
            did = int(did) - 1

            if qid not in relevance:
                relevance[qid] = []

            relevance[qid].append(did)

    return relevance


def precision_at_k(retrieved, relevant, k=10):

    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = len(set(retrieved_k) & relevant_set)

    return hits / k


def recall_at_k(retrieved, relevant, k=10):

    if len(relevant) == 0:
        return 0

    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = len(set(retrieved_k) & relevant_set)

    return hits / len(relevant)