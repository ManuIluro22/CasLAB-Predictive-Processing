import numpy as np
import pandas as pd
from scipy import stats
import random
from itertools import product

def generate_initial_combinations(features, num_combinations=50):
    percentiles = list(range(10, 90, 5))  # Adjusted range and step
    all_combinations = list(product(features, percentiles))
    random.shuffle(all_combinations)
    return [{'feature': combo[0], 'percentile': combo[1]} for combo in all_combinations[:num_combinations]]

def test_significance(metrics, group1, group2, p_values):
    """ Helper function to test significance and return p-value """
    for metric in metrics:
        if not group1[metric].empty and not group2[metric].empty:
            u_stat, p_value = stats.mannwhitneyu(group1[metric], group2[metric], alternative='two-sided')
            if p_value < 0.05:
                p_values.append(p_value)
    return len(p_values) > 0  # True if any significant differences found

def initial_iteration(initializations, feature_scales, metrics_columns):
    results = []
    for init in initializations:
        feature = init['feature']
        percentile_cutoff = init['percentile']
        p_values = []

        percentile = np.percentile(feature_scales[feature], percentile_cutoff)
        above_values = feature_scales[feature_scales[feature] > percentile]
        below_values = feature_scales[feature_scales[feature] <= percentile]

        for metric in metrics_columns:
            u_stat, p_value = stats.mannwhitneyu(below_values[metric], above_values[metric], alternative='two-sided')
            if p_value < 0.05:
                p_values.append(p_value)

        if p_values:
            mean_p_value = np.mean(p_values)
            results.append(({
                'initial_feature': feature,
                'initial_percentile': percentile_cutoff,
                'significant_counts': len(p_values),
                'mean_p_value': mean_p_value,
                'lineage': [[{'feature': feature, 'percentile': percentile_cutoff}]]
            },{
                'A': above_values,
                'B': below_values
            }))
    results.sort(key=lambda x: (-x[0]['significant_counts'], x[0]['mean_p_value']))
    return results

def perform_iterations(previous_results, feature_scales, metrics_columns, features, add_count = False, num_combinations = 100, min_samples = 5,min_split = 15, num_iterations=1):
    if num_iterations == 0:
        return previous_results

    extended_results = []
    for result, clusters in previous_results:

        for combo in generate_initial_combinations([f for f in features if f != result['initial_feature']], num_combinations):
            new_feature = combo['feature']
            new_percentile = combo['percentile']
            lineage = result['lineage'].copy()
            lineage.append([{'feature': new_feature, 'percentile': new_percentile}])
            new_value = np.percentile(feature_scales[new_feature], new_percentile)

            final_clusters = {}
            p_values = []

            for key, data in clusters.items():
                if len(data) <= min_split:
                    final_clusters[key] = data
                    continue

                new_above = data[data[new_feature] > new_value]
                new_below = data[data[new_feature] <= new_value]
                if (len(new_above) < min_samples or len(new_below) < min_samples):
                    final_clusters[key] = data
                    continue

                if test_significance(metrics_columns, new_above, new_below, p_values):
                    final_clusters[key + 'A'] = new_above
                    final_clusters[key + 'B'] = new_below
                else:
                    final_clusters[key] = data
            if add_count:
                count = len(p_values) + result["significant_counts"]
            else:
                count = len(p_values)
            if p_values:
                mean_p_value = np.mean(p_values)
                extended_results.append(({
                    'initial_feature': result['initial_feature'],
                    'initial_percentile': result['initial_percentile'],
                    'new_feature': new_feature,
                    'new_percentile': new_percentile,
                    'significant_counts': count,
                    'mean_p_value': mean_p_value,
                    'lineage': lineage
                }, final_clusters))

        extended_results.sort(key=lambda x: (-x[0]['significant_counts'], x[0]['mean_p_value']))

    return perform_iterations(extended_results[:10], feature_scales, metrics_columns, features, add_count, num_combinations, min_samples,min_split, num_iterations - 1)


def analyze_top_configurations(configurations, feature_scales, metrics_columns, top_n=5):
    # Analyze each configuration in the top 'top_n'
    for init in configurations[:top_n]:
        initial_feature = init[0]['initial_feature']
        initial_percentile = init[0]['initial_percentile']
        new_feature = init[0]['new_feature']
        new_percentile = init[0]['new_percentile']

        print(f"\nConfiguration: Initial Feature={initial_feature}, Percentile={initial_percentile}, "
              f"New Feature={new_feature}, New Percentile={new_percentile}")

        clusters = init[1]

        cluster_sizes = {key: len(feature_scales.loc[clusters[key].index]) for key in clusters}
        print("Cluster Sizes:")
        for key, size in cluster_sizes.items():
            print(f"  {key}: {size}")

        metric_means = {metric: {key: feature_scales.loc[clusters[key].index][metric].mean() for key in clusters if
                                 not feature_scales.loc[clusters[key].index][metric].empty} for metric in
                        metrics_columns}

        print("Metric Means Across Clusters:")
        for metric, means in metric_means.items():
            mean_values = ", ".join(f"{key}: {means[key]:.4f}" for key in sorted(means))
            print(f"  {metric}: {mean_values}")