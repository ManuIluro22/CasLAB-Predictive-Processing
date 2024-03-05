from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering, KMeans, OPTICS
from dtaidistance import dtw
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def hierarchical_clustering_ts(list_df,metric,params,plotting = True):
    dist_mat_list = []

    for df in list_df:
        ts_data= df.astype(np.float64)
        distance_matrix = metric(ts_data)
        dist_mat_list.append(distance_matrix.astype(np.float64))
    if len(dist_mat_list) == 1:
        condensed_dist_matrix = squareform(dist_mat_list[0])
    else:
        condensed_dist_matrix = squareform(np.mean(dist_mat_list,axis=0))


    # Ensure params is a dictionary, even if None
    Z = linkage(condensed_dist_matrix, **params)

    if plotting == True:
        #plot_time_series(dendrogram(Z),title = "Hierarchical Clustering Dendrogram")
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Time Series Index')
        plt.ylabel('Distance')
        plt.show()
        # %%
    return Z



def create_cluster_df(linkage,df,params):
    clusters_df = fcluster(linkage, **params)
    df.reset_index(inplace=True)
    df.loc[:, "clusters"] = clusters_df.copy()

    return df[["Subject ID", "clusters"]].copy()
