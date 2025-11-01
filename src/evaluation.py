# Module for evaluating clustering performance

from sklearn.metrics import silhouette_score

def evaluate_clusters(data, labels):
    # assessing clustering quality using Silhouette Score
    score = silhouette_score(data, labels)
    print(f"Silhouette Score: {score:.4f}")
    return score
