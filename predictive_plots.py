import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_boxplot(melted_df, variable, save_plot=False, index=None):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Plot boxplot for the current variable
    sns.boxplot(x='clusters', y='Mean_Score', data=melted_df, ax=ax)
    ax.set_title(f'Boxplot for {variable}')
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Mean Score')
    ax.grid(True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Calculate mean
    means = melted_df[melted_df['Variable'] == variable].groupby('clusters')['Mean_Score'].mean()

    for cluster, mean in enumerate(means):
        ax.text(cluster, mean, f'{mean:.2f}', ha='center', va='top', color='red')

    plt.tight_layout()
    if save_plot == True:
        plot_filename = f'plot_{index[0]}_{index[1]}.png'  # Generate unique filename
        plt.savefig(plot_filename)
        plt.close()
        return plot_filename
    else:
        plt.show()


def create_stats_table(melted_df, variable, size_clust, save_plot=False, index=None):
    stats = melted_df[melted_df['Variable'] == variable].groupby('clusters')['Mean_Score'].agg(['size', 'mean', 'std'])

    stats.reset_index(inplace=True)
    stats["clusters"] = stats["clusters"]
    stats["mean"] = stats["mean"].round(2)
    stats["std"] = stats["std"].round(2)
    stats['SE of Cluster'] = (stats['std'] / np.sqrt(stats['size'])).round(2)
    stats['size'] = size_clust

    stats.columns = ["Number of Cluster", "Size of Cluster", "Mean of Cluster", "Standard Deviation of Cluster",
                     "SE of Cluster"]

    plt.figure(figsize=(8, 2))
    table = plt.table(cellText=stats.drop("Standard Deviation of Cluster", axis=1).values,
                      colLabels=stats.drop("Standard Deviation of Cluster", axis=1).columns,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1)
    table.auto_set_column_width([i for i in range(len(stats.columns))])
    plt.axis('off')
    plt.title(f'Statistics for {variable}')
    if save_plot == True:
        table_filename = f'table_{index[0]}_{index[1]}.png'  # Generate unique filename
        plt.savefig(table_filename)
        plt.close()
        return table_filename
    else:
        plt.show()