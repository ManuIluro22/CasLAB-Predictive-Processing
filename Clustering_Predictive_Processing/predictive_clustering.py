from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering, KMeans, OPTICS
from dtaidistance import dtw
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def hierarchical_clustering_ts(list_df, metric, params, plotting=True):
    """
    Performs hierarchical clustering on time series data and optionally plots the dendrogram.

    Parameters:
    - list_df: A list of DataFrames containing time series data.
    - metric: A function to compute the distance matrix from time series data.
    - params: Parameters for the linkage method used in hierarchical clustering.
    - plotting: Boolean indicating whether to plot the dendrogram.

    Returns:
    - Z: The hierarchical clustering encoded as a linkage matrix.
    """

    dist_mat_list = []

    # Compute distance matrices for each DataFrame in list_df
    for df in list_df:
        ts_data = df.astype(np.float64)  # Ensure data is in float64 format
        distance_matrix = metric(ts_data)  # Compute distance matrix
        dist_mat_list.append(distance_matrix.astype(np.float64))  # Store distance matrix

    # Combine distance matrices if there are multiple, otherwise use the single matrix
    if len(dist_mat_list) == 1:
        condensed_dist_matrix = squareform(dist_mat_list[0])
    else:
        # Average distance matrices if there's more than one
        condensed_dist_matrix = squareform(np.mean(dist_mat_list, axis=0))

    # Perform hierarchical clustering
    Z = linkage(condensed_dist_matrix, **params)

    # Plot dendrogram if plotting is enabled
    if plotting:
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Time Series Index')
        plt.ylabel('Distance')
        plt.show()

    return Z


def create_cluster_df(linkage, df, params):
    """
    Creates a DataFrame with 'Subject ID' and cluster assignments obtained from hierarchical clustering.

    Parameters:
    - linkage: Linkage matrix from hierarchical clustering.
    - df: Original DataFrame containing 'Subject ID'.
    - params: Parameters for forming flat clusters.

    Returns:
    - DataFrame with 'Subject ID' and corresponding cluster assignments.
    """

    # Form flat clusters from the hierarchical clustering
    clusters_df = fcluster(linkage, **params)
    df.reset_index(inplace=True)  # Reset index of the original DataFrame
    df.loc[:, "clusters"] = clusters_df.copy()  # Assign cluster labels to 'clusters' column

    return df[["Subject ID", "clusters"]].copy()  # Return DataFrame with 'Subject ID' and 'clusters'


def length_cluster(df_cluster):
    """
    Counts the number of occurrences for each cluster.

    Parameters:
    - df_cluster: DataFrame containing cluster assignments.

    Returns:
    - DataFrame with each cluster and its corresponding count.
    """

    # Count occurrences of each cluster
    unique_clusters, counts = np.unique(df_cluster["clusters"], return_counts=True)
    cluster_counts_df = pd.DataFrame({
        'Cluster': unique_clusters,
        'Count': counts
    })

    return cluster_counts_df
