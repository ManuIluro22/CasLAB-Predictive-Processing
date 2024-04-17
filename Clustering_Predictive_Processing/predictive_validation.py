
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from scipy.optimize import linear_sum_assignment

def analyze_cluster_stability(df_cluster, scaled_df, cluster_indices_list, n_clusters, output_list):
    """
       Analyzes the stability of cluster assignments in a dataset by removing each subject
       one at a time, re-clustering the remaining data, and then comparing the new cluster
       assignments to the original using the Jaccard Index. This method is also known as
       the "leave-one-out" cluster stability analysis.

       Parameters:
       - df_cluster: DataFrame containing the original cluster assignments with a 'clusters' column.
       - scaled_df: DataFrame containing the scaled data used for clustering.
       - cluster_indices_list: List containing the indices of the clusters to analyze.
       - n_clusters: The number of clusters used in the original clustering.
       - output_list: List to append the results of the analysis for each left-out subject.

       Returns:
       - Returns the modified `output_list` with appended results, where each result includes
         the belonging cluster of the left-out subject, the original cluster, the new cluster after re-clustering,
         the index of the left-out subject, and the computed Jaccard Index for the matched clusters.

       Note: This function assumes the presence of a 'clusters' column in `df_cluster` and uses
       AgglomerativeClustering with 'complete' linkage for re-clustering. It employs a cost matrix
       to align new clusters with original clusters for comparison, aiming to assess the impact
       of removing each subject on the overall cluster configuration.
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


def bootstrap_validation(n_samples, n_clusters, consistency_results, means_list, original_clusters, original_means_df
                         ,list_metrics,scaled_df,df,scales):
    """
    Performs a bootstrap validation procedure by repeatedly sampling with replacement, clustering the samples, and
    comparing the consistency and differences in means of cluster assignments against original clusters.

    Parameters:
        - n_samples: The number of bootstrap samples to generate.
        - n_clusters: The number of clusters to form in each bootstrap sample.
        - consistency_results: List to store consistency results for each bootstrap iteration.
        - means_list: List to store differences in means for metrics between original and bootstrap clusters.
        - original_clusters: List of arrays containing the indices of original cluster members.
        - original_means_df: DataFrame containing the mean values of metrics for original clusters.
        - list_metrics: List of lists, where each sublist contains metric names to evaluate.
        - scaled_df: DataFrame with scaled data used for clustering.
        - df: Original DataFrame from which additional attributes might be required.
        - scales: DataFrame containing the metric values with 'EPRIME_CODE' as a key.

    Returns:
        None, but modifies the `consistency_results` and `means_list` in place to include results from each bootstrap sample.

    Note: This function employs an AgglomerativeClustering model with 'complete' linkage. It assumes the presence of a 'Subject' column in `df` and 'EPRIME_CODE' in `scales`. It calculates the consistency and mean difference
    of cluster assignments against original clusters for specified metrics.
    """
    for n in range(n_samples):
        # Generate a bootstrap sample of the indices
        sample_indices = np.random.choice(scaled_df.index, size=len(scaled_df), replace=True)
        sample_df = scaled_df.loc[sample_indices]
        # Cluster the bootstrap sample
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete")
        sample_clusters = cluster_model.fit_predict(sample_df)

        # Reindex sample_clusters to align with original data indices
        sample_cluster_assignments = pd.Series(sample_clusters, index=sample_indices)

        cost_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                # Intersection size between original cluster i and sample cluster j
                orig_indices = original_clusters[i]
                sample_indices_in_j = sample_cluster_assignments[sample_cluster_assignments == j].index
                intersection = len(set(orig_indices) & set(sample_indices_in_j))
                cost_matrix[i, j] = -intersection  # Negative because we use minimum cost matching

        # Match original clusters to sample clusters
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_clusters = dict(zip(row_ind, col_ind))

        # Calculate consistency
        for i in range(n_clusters):
            j = matched_clusters[i]
            orig_indices = set(original_clusters[i])
            sample_indices_in_j = set(sample_cluster_assignments[sample_cluster_assignments == j].index)
            consistency = len(orig_indices & sample_indices_in_j) / len(sample_indices_in_j)
            consistency_results.append({'OriginalCluster': i, 'Sample': n, 'Consistency': consistency})

        df_sample_cluster_assignments = sample_cluster_assignments.reset_index()
        df_sample_cluster_assignments.columns = ["index", "clusters"]

        subject_cluster = df.loc[sample_indices]["Subject"].reset_index().merge(df_sample_cluster_assignments,
                                                                                      left_index=True,
                                                                                      right_index=True).drop(
            ["index_x", "index_y"], axis=1)
        scales_cluster = pd.merge(scales, subject_cluster, left_on='EPRIME_CODE', right_on='Subject')
        scales_cluster.set_index(sample_indices, inplace=True)

        for metrics in list_metrics:
            for group in metrics:
                scores_cluster = scales_cluster[[group, "clusters"]]
                means = scores_cluster.groupby("clusters").mean()
                means.index = col_ind

                for index, row in means.iterrows():
                    means_list.append({
                        'OriginalCluster': row.name,
                        'Metric': group,
                        'Dif_Mean': original_means_df[(original_means_df['OriginalCluster'] == row.name) & (
                                    original_means_df['Metric'] == group)]["Mean"].values[0] - row.values[0]
                    })
