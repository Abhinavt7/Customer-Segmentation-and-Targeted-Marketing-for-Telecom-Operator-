# Clustering Algorithms for Customer Segmentation

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_optimal_clusters(data, k_range):
    # Finding the optimal number of clusters using the Elbow Method
    from sklearn.metrics import silhouette_score
    
    sse = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, max_iter=100)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        
        if k > 1:
            score = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(score)
            print(f"Silhouette score for k={k}: {score:.4f}")
    
    # Plotting elbow
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE (Inertia)')
    plt.title('Elbow Method')
    plt.savefig('../visualizations/elbow_plot.png')
    plt.show()
    
    return sse, silhouette_scores

def train_kmeans(data, n_clusters):
    # Training KMeans clustering model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=100)
    kmeans.fit(data)
    return kmeans
