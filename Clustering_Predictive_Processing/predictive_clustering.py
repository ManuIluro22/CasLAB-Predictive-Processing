from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering, KMeans, OPTICS
from dtaidistance import dtw
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score
from scipy.optimize import linear_sum_assignment



def clustering(df,algorithm, params,fit=False):
    """
    Creates a DataFrame with 'Subject ID' and cluster assignments obtained from the given clustering algorithm.

    Parameters:
    - df: Original DataFrame containing 'Subject ID'.
    - algorithm: Function/algorithm that performs the clustering
    - params: Parameters for forming flat clusters.

    Returns:
    - DataFrame with 'Subject ID' and corresponding cluster assignments.
    """

    # Form flat clusters from the hierarchical clustering
    if fit:
        clustering= algorithm(**params).fit(df)

        return clustering.labels_
    
    else:
        clusters_df = algorithm(**params)
        df.reset_index(inplace=True)  # Reset index of the original DataFrame
        df.loc[:, "clusters"] = clusters_df.copy()  # Assign cluster labels to 'clusters' column

        return df[["Subject ID", "clusters"]].copy()  # Return DataFrame with 'Subject ID' and 'clusters'

def hierarchical_clustering_ts_linkage(list_df, metric, params, plotting=True):
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


def analyze_cluster_stability(df_cluster, scaled_df, cluster_indices_list, n_clusters, output_list):
    """
    Analyzes the stability of clustering by re-clustering with one subject left out
    and calculating the Jaccard index for the reformed clusters compared to the original.

    Args:
        df_cluster (DataFrame): DataFrame containing the original cluster assignments.
        scaled_df (DataFrame): DataFrame containing scaled features for clustering.
        cluster_indices_list (list): List of cluster indices to analyze.
        n_clusters (int): Number of clusters for re-clustering.
        output_list (list): List to which results will be appended.

    Returns:
        DataFrame: A DataFrame containing the results of the stability analysis.
    """
    # Iterate over each cluster
    for cluster in cluster_indices_list:
        cluster_indices = df_cluster[df_cluster['clusters'] == cluster].index
        for idx in cluster_indices:
            temp_df = scaled_df.drop(idx)
            cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete")
            new_clusters = cluster_model.fit_predict(temp_df)

            # Reconstruct the full cluster assignment series including the removed subject as NaN
            new_cluster_assignments = pd.Series(index=scaled_df.index, dtype='float64')
            new_cluster_assignments.iloc[temp_df.index] = new_clusters

            # Calculate cost matrix for cluster matching
            cost_matrix = np.zeros((n_clusters, n_clusters))
            for i in range(n_clusters):
                original_in_cluster = (df_cluster['clusters'] == i).to_numpy()
                for j in range(n_clusters):
                    new_in_cluster = (new_cluster_assignments == j).to_numpy()
                    # Using negative intersection size as cost because we want maximum matching
                    cost_matrix[i, j] = -np.sum(original_in_cluster & new_in_cluster)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # For each cluster, calculate Jaccard Index using matched clusters
            for i, j in zip(row_ind, col_ind):
                orig_labels = (df_cluster['clusters'] == i) & (df_cluster.index != idx)
                new_labels = (new_cluster_assignments == j) & (new_cluster_assignments.index != idx)

                # Calculate the Jaccard Index for matched clusters
                jaccard_index = jaccard_score(orig_labels, new_labels)

                output_list.append({
                    'BelongingCluster': cluster,
                    'OriginalCluster': i,
                    'NewCluster': j,
                    'LeftOutSubjectIndex': idx,
                    'Jaccard_Index': round(jaccard_index, 2)
                })

    return output_list

# %%
