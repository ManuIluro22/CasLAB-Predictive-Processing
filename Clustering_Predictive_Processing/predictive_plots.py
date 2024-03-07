import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_stats_table(melted_df, variable, size_clust, save_plot=False, index=None):
    """
    Creates a statistics table for a specified variable within a melted DataFrame.
    The table includes cluster size, mean, standard deviation, and standard error.

    Parameters:
    - melted_df: DataFrame containing melted data.
    - variable: The variable for which statistics are calculated.
    - size_clust: An array of cluster sizes.
    - save_plot: Boolean indicating whether to save the table as an image.
    - index: Tuple or list with two elements used to generate a unique filename.

    Returns:
    - Filename of the saved table image if save_plot is True. Otherwise, displays the table.
    """

    # Calculate size, mean, and standard deviation for each cluster
    stats = melted_df[melted_df['Variable'] == variable].groupby('clusters')['Mean_Score'].agg(['size', 'mean', 'std'])

    # Reset index to turn indices into columns
    stats.reset_index(inplace=True)
    # Round mean and standard deviation for readability
    stats["mean"] = stats["mean"].round(2)
    stats["std"] = stats["std"].round(2)
    # Calculate standard error of mean (SE) and round it
    stats['SE of Cluster'] = (stats['std'] / np.sqrt(stats['size'])).round(2)
    # Replace size with actual cluster sizes
    stats['size'] = size_clust

    # Rename columns for clarity
    stats.columns = ["Number of Cluster", "Size of Cluster", "Mean of Cluster", "Standard Deviation of Cluster",
                     "SE of Cluster"]

    # Initialize figure for plotting
    plt.figure(figsize=(8, 2))
    # Create the table and adjust its properties
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

    # Save or display the table based on the save_plot flag
    if save_plot:
        table_filename = f'table_{index[0]}_{index[1]}.png' if index else 'table.png'  # Provide fallback filename
        plt.savefig(table_filename)
        plt.close()
        return table_filename
    else:
        plt.show()

def create_boxplot(melted_df, variable, save_plot=False, index=None):
    """
    Creates a boxplot for a specified variable within a melted DataFrame.

    Parameters:
    - melted_df: DataFrame containing melted data for plotting.
    - variable: The variable to plot in the boxplot.
    - save_plot: Boolean indicating whether to save the plot as an image.
    - index: Tuple or list with two elements used to generate a unique filename.

    Returns:
    - Filename of the saved plot image if save_plot is True. Otherwise, displays the plot.
    """

    # Initialize figure and axis for the plot
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Create the boxplot
    sns.boxplot(x='clusters', y='Mean_Score', data=melted_df, ax=ax)
    ax.set_title(f'Boxplot for {variable}')
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Mean Score')
    ax.grid(True)

    # Rotate x-tick labels for better readability
    plt.xticks(rotation=45)

    # Calculate and annotate means for each cluster
    means = melted_df[melted_df['Variable'] == variable].groupby('clusters')['Mean_Score'].mean()
    for cluster, mean in means.items():
        cluster_position = list(means.keys()).index(cluster)
        ax.text(cluster_position, mean, f'{mean:.2f}', ha='center', va='top', color='red')

    plt.tight_layout()

    # Save or display the plot based on the save_plot flag
    if save_plot:
        plot_filename = f'plot_{index[0]}_{index[1]}.png' if index else 'plot.png'  # Provide fallback filename
        plt.savefig(plot_filename)
        plt.close()
        return plot_filename
    else:
        plt.show()

def create_boxplot_emotions(df, df_cluster):
    """
    Creates boxplots for emotion scores grouped by clusters. The function
    calculates mean scores for 'No Match' and 'Match' conditions across
    three emotions: Happy, Sad, and Fear.

    Parameters:
    - df: DataFrame containing the raw scores for each emotion and condition.
    - df_cluster: DataFrame containing cluster assignments for each subject.

    No return value; plots are displayed directly.
    """

    # Copy the original DataFrame and merge cluster information
    copy_data = df.copy()
    copy_data["clusters"] = df_cluster["clusters"]

    # Calculate mean scores for 'No Match' and 'Match' conditions for each emotion
    copy_data['Happy_No_Match'] = copy_data[
        ['Happy_0_0', 'Happy_0_1', 'Happy_0_2', 'Happy_0_3', 'Happy_0_4', 'Happy_0_5']].mean(axis=1)
    copy_data['Happy_Match'] = copy_data[
        ['Happy_1_0', 'Happy_1_1', 'Happy_1_2', 'Happy_1_3', 'Happy_1_4', 'Happy_1_5', 'Happy_1_6', 'Happy_1_7',
         'Happy_1_8']].mean(axis=1)

    # Repeat for 'Sad' and 'Fear' emotions
    copy_data['Sad_No_Match'] = copy_data[['Sad_0_0', 'Sad_0_1', 'Sad_0_2', 'Sad_0_3', 'Sad_0_4', 'Sad_0_5']].mean(axis=1)
    copy_data['Sad_Match'] = copy_data[
        ['Sad_1_0', 'Sad_1_1', 'Sad_1_2', 'Sad_1_3', 'Sad_1_4', 'Sad_1_5', 'Sad_1_6', 'Sad_1_7', 'Sad_1_8']].mean(axis=1)

    copy_data['Fear_No_Match'] = copy_data[
        ['Fear_0_0', 'Fear_0_1', 'Fear_0_2', 'Fear_0_3', 'Fear_0_4', 'Fear_0_5']].mean(axis=1)
    copy_data['Fear_Match'] = copy_data[
        ['Fear_1_0', 'Fear_1_1', 'Fear_1_2', 'Fear_1_3', 'Fear_1_4', 'Fear_1_5', 'Fear_1_6', 'Fear_1_7',
         'Fear_1_8']].mean(axis=1)

    # Melt the DataFrame for boxplot visualization
    happy_df = pd.melt(copy_data, id_vars=['clusters'], value_vars=['Happy_No_Match', 'Happy_Match'],
                       var_name='Happy', value_name='Mean_Score')
    sad_df = pd.melt(copy_data, id_vars=['clusters'], value_vars=['Sad_No_Match', 'Sad_Match'],
                     var_name='Sad', value_name='Mean_Score')
    fear_df = pd.melt(copy_data, id_vars=['clusters'], value_vars=['Fear_No_Match', 'Fear_Match'],
                      var_name='Fear', value_name='Mean_Score')

    # Plot boxplots for each emotion separately
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    sns.boxplot(x='Happy', y='Mean_Score', hue='clusters', data=happy_df)
    plt.title('Boxplot of Mean Happy Scores for Each Cluster')

    plt.subplot(1, 3, 2)
    sns.boxplot(x='Sad', y='Mean_Score', hue='clusters', data=sad_df)
    plt.title('Boxplot of Mean Sad Scores for Each Cluster')

    plt.subplot(1, 3, 3)
    sns.boxplot(x='Fear', y='Mean_Score', hue='clusters', data=fear_df)
    plt.title('Boxplot of Mean Fear Scores for Each Cluster')

    plt.tight_layout()
    plt.show()

