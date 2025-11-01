# visualization functions for clustering results
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_elbow(sse, k_range):
    # Elbow plot for SSE vs. number of clusters
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, sse, marker='o', color='green')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (SSE)')
    plt.title('Elbow Method')
    plt.savefig('../visualizations/elbow_plot.png')
    plt.show()

def plot_silhouette(silhouette_scores, k_range):
    #silhouette score plot
    plt.figure(figsize=(10, 5))
    plt.plot(k_range[1:], silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.savefig('../visualizations/silhouette_plot.png')
    plt.show()

def plot_cluster_profiles(data, numerical_cols):
    #cluster profile plots
    for col in numerical_cols:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='Cluster', y=col, data=data, palette='Set2')
        plt.title(f'Distribution of {col} by Cluster')
        plt.savefig(f'../visualizations/cluster_{col}.png')
        plt.show()

def plot_cluster_radar(data, cluster_col, features, normalize=True):
    
    # cluster_col -> str, Column name containing cluster labels.
    # features -> list, List of numeric features to include in the radar plot.    
    # normalize -> bool, (if needed) If True, normalize features to [0,1] for comparability.
        

    # Compute cluster means
    cluster_means = data.groupby(cluster_col)[features].mean()

    # Normalize if requested

    # min_ = cluster_means.min()
    min_ = 0

    if normalize:
        cluster_means = (cluster_means - min_ ) / (cluster_means.max() - min_)#cluster_means.min())

    # Number of variables
    categories = list(cluster_means.columns)
    N = len(categories)

    # Angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the loop

    # Plot each cluster
    plt.figure(figsize=(8, 8))
    for idx, row in cluster_means.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]  # close the loop
        plt.polar(angles, values, label=f'Cluster {idx}', linewidth=2)
        plt.fill(angles, values, alpha=0.1)

    # Add labels
    plt.xticks(angles[:-1], categories, fontsize=10)
    plt.yticks([0.25, 0.5, 0.75], ["0.25","0.5","0.75"], color="grey", size=8)
    plt.title("Cluster Profiles (Radar Plot)", size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.savefig(f'../visualizations/cluster_radar.png',bbox_inches='tight')
    plt.show()
