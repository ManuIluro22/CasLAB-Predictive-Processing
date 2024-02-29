import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fcluster

def create_cluster_df(linkage,n_clusters,df):
    clusters_happy_match = fcluster(linkage, n_clusters, criterion='maxclust')
    df.reset_index(inplace=True)
    df.loc[:, "clusters"] = clusters_happy_match
    new_df = df[["Subject ID", "clusters"]].copy()
    return new_df

def create_metrics_cluster_df(df_cluster, df_scales):
    new_data = pd.merge(df_cluster, df_scales, left_on='Subject ID', right_on='EPRIME_CODE')

    filter_df = filter_data(new_data)

    data_clust = {}
    for clust in np.unique(filter_df["clusters"]):
        metric_data = {}
        for column in filter_df.drop(["EPRIME_CODE", "clusters"], axis=1).columns:
            mean_col = filter_df[filter_df["clusters"] == clust][column].mean()
            std_col = filter_df[filter_df["clusters"] == clust][column].std()
            max_col = filter_df[filter_df["clusters"] == clust][column].max()
            min_col = filter_df[filter_df["clusters"] == clust][column].min()

            # Normality Test
            # _, p_value_col = stats.shapiro(result_df[result_df["clusters"]==clust][column])
            metric_data[column] = {'mean': round(mean_col, 2), 'std': round(std_col, 2),
                                   'max': max_col, 'min': min_col}

        data_clust[clust] = metric_data

    df_clusters = pd.DataFrame()
    for cluster, attributes in data_clust.items():
        for attribute, values in attributes.items():
            for stat, value in values.items():
                col_name = f"{attribute}_{stat}"
                df_clusters.loc[cluster, col_name] = value

    return df_clusters



def filter_data(df,data_type = np.number,na_number = 20):
    numeric_columns = df.select_dtypes(include=[data_type]).columns
    columns_to_select = ['EPRIME_CODE'] + list(numeric_columns)

    # Selecting all numeric columns along with "Subject"
    filtered_df = df[columns_to_select]
    cols_to_drop = filtered_df.columns[filtered_df.isnull().sum() > na_number]
    filtered_df = filtered_df.drop(columns=cols_to_drop).copy()

    return filtered_df.drop("Age", axis=1, inplace=True)


def create_metrics_cluster_df(df_cluster, df_scales):
    new_data = pd.merge(df_cluster, df_scales, left_on='Subject ID', right_on='EPRIME_CODE')

    filter_df = filter_data(new_data)

    data_clust = {}
    for clust in np.unique(filter_df["clusters"]):
        metric_data = {}
        for column in filter_df.drop(["EPRIME_CODE", "clusters"], axis=1).columns:
            mean_col = filter_df[filter_df["clusters"] == clust][column].mean()
            # std_col = filter_df[filter_df["clusters"] == clust][column].std()
            # max_col = filter_df[filter_df["clusters"] == clust][column].max()
            # min_col = filter_df[filter_df["clusters"] == clust][column].min()

            # Normality Test
            # _, p_value_col = stats.shapiro(result_df[result_df["clusters"]==clust][column])
            metric_data[column] = {'mean': round(mean_col, 2)}  # , 'std': round(std_col, 2),
            # 'max': max_col, 'min': min_col}

        data_clust[clust] = metric_data

    df_clusters = pd.DataFrame()
    for cluster, attributes in data_clust.items():
        for attribute, values in attributes.items():
            for stat, value in values.items():
                col_name = f"{attribute}_{stat}"
                df_clusters.loc[cluster, col_name] = value

    return df_clusters


def filter_data(df, data_type=np.number, na_number=60):
    numeric_columns = df.select_dtypes(include=[data_type]).columns
    columns_to_select = ['EPRIME_CODE'] + list(numeric_columns)

    # Selecting all numeric columns along with "Subject"
    filtered_df = df[columns_to_select]
    cols_to_drop = filtered_df.columns[filtered_df.isnull().sum() >= na_number]
    filtered_df = filtered_df.drop(columns=cols_to_drop).copy()
    return filtered_df.drop("Age", axis=1)


def create_mean_tasks(df, df_cluster):
    copy_data = df.copy()
    copy_data["clusters"] = df_cluster["clusters"]

    # Extract cluster labels
    clusters = copy_data['clusters'].unique()

    # Create an empty DataFrame to store the mean scores
    df_mean_scores = pd.DataFrame(columns=['Cluster', 'Happy_0', 'Happy_1', 'Sad_0', 'Sad_1', 'Fear_0', 'Fear_1'])

    # Iterate over each cluster
    for cluster in clusters:
        # Filter data for the current cluster
        cluster_data = copy_data[copy_data['clusters'] == cluster]

        # Calculate mean scores for each category
        df_mean_scores.loc[len(df_mean_scores)] = [
            cluster,
            cluster_data[['Happy_0_' + str(i) for i in range(6)]].mean().mean(),
            cluster_data[['Happy_1_' + str(i) for i in range(9)]].mean().mean(),
            cluster_data[['Sad_0_' + str(i) for i in range(6)]].mean().mean(),
            cluster_data[['Sad_1_' + str(i) for i in range(9)]].mean().mean(),
            cluster_data[['Fear_0_' + str(i) for i in range(6)]].mean().mean(),
            cluster_data[['Fear_1_' + str(i) for i in range(9)]].mean().mean()
        ]

    # Display the mean scores for each cluster
    return df_mean_scores


def create_boxplots(df, df_cluster):
    copy_data = df.copy()
    copy_data["clusters"] = df_cluster["clusters"]

    # Compute mean for each subject separately for 0 and 1
    copy_data['Happy_No_Match'] = copy_data[
        ['Happy_0_0', 'Happy_0_1', 'Happy_0_2', 'Happy_0_3', 'Happy_0_4', 'Happy_0_5']].mean(axis=1)
    copy_data['Happy_Match'] = copy_data[
        ['Happy_1_0', 'Happy_1_1', 'Happy_1_2', 'Happy_1_3', 'Happy_1_4', 'Happy_1_5', 'Happy_1_6', 'Happy_1_7',
         'Happy_1_8']].mean(axis=1)

    copy_data['Sad_No_Match'] = copy_data[['Sad_0_0', 'Sad_0_1', 'Sad_0_2', 'Sad_0_3', 'Sad_0_4', 'Sad_0_5']].mean(
        axis=1)
    copy_data['Sad_Match'] = copy_data[
        ['Sad_1_0', 'Sad_1_1', 'Sad_1_2', 'Sad_1_3', 'Sad_1_4', 'Sad_1_5', 'Sad_1_6', 'Sad_1_7', 'Sad_1_8']].mean(
        axis=1)

    copy_data['Fear_No_Match'] = copy_data[
        ['Fear_0_0', 'Fear_0_1', 'Fear_0_2', 'Fear_0_3', 'Fear_0_4', 'Fear_0_5']].mean(axis=1)
    copy_data['Fear_Match'] = copy_data[
        ['Fear_1_0', 'Fear_1_1', 'Fear_1_2', 'Fear_1_3', 'Fear_1_4', 'Fear_1_5', 'Fear_1_6', 'Fear_1_7',
         'Fear_1_8']].mean(axis=1)

    # Melt the DataFrame to long format for each emotion
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
    plt.xlabel('Happy')
    plt.ylabel('Mean Score')
    plt.legend(title='Cluster')

    plt.subplot(1, 3, 2)
    sns.boxplot(x='Sad', y='Mean_Score', hue='clusters', data=sad_df)
    plt.title('Boxplot of Mean Sad Scores for Each Cluster')
    plt.xlabel('Sad')
    plt.ylabel('Mean Score')
    plt.legend(title='Cluster')

    plt.subplot(1, 3, 3)
    sns.boxplot(x='Fear', y='Mean_Score', hue='clusters', data=fear_df)
    plt.title('Boxplot of Mean Fear Scores for Each Cluster')
    plt.xlabel('Fear')
    plt.ylabel('Mean Score')
    plt.legend(title='Cluster')

    plt.tight_layout()
    plt.show()
