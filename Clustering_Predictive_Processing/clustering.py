from sklearn.ensemble import RandomTreesEmbedding
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import seaborn as sns
import matplotlib.pyplot as plt


def hierarchical_forest_clustering(attributes, df, visualize=True):
    embedder = RandomTreesEmbedding(n_estimators=100, random_state=42)

    X_transformed = embedder.fit_transform(df[attributes])
    # Compute Proximity Matrix
    proximity_matrix = (X_transformed @ X_transformed.T).toarray()

    # Normalize Proximity Matrix
    proximity_matrix = proximity_matrix / proximity_matrix.max()

    # Clustering
    linked = linkage(proximity_matrix, method='single')
    dendrogram(linked, labels=df.index.tolist())
    if visualize:
        plt.show()
    return linked


def determine_clusters(linked, n=5):
    clusters = fcluster(linked, n, criterion='maxclust')
    return clusters

def create_and_plot_clusters(data,attributes,n_clusters=5,x="NA",y="PA"):
    link = hierarchical_forest_clustering(attributes,data,visualize=True)
    data['cluster'] = determine_clusters(linked=link,n=n_clusters)
    sns.scatterplot(data=data, x = x, y = y, hue= 'cluster',palette="deep")
    plt.plot
    return data
