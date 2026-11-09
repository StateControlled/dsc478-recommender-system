# data/processed/
# user_vectors.csv

# compiled_data.csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import sys
from IPython.core.display_functions import display

RANDOM_STATE=42
NUM_CLUSTERS = 4
DO_RANGE = False

def cluster_sizes(clusters, data):
    # clusters is an array of cluster labels for each instance in the data

    size = {}
    cluster_labels = np.unique(clusters)

    for c in cluster_labels:
        size[c] = len(data[clusters == c])
    return size


def run():
    data: pd.DataFrame = pd.read_csv("../data/processed/user_vectors.csv", index_col=0)



    # users x books
    display(data.head(10))

    kmeans = KMeans(n_clusters=NUM_CLUSTERS, max_iter=200, init="k-means++", n_init=2, verbose=1, random_state=RANDOM_STATE)
    kmeans.fit(data)

    #######################################################

    print("Clusters")
    clusters = kmeans.predict(data)
    print(clusters)
    print()

    #######################################################

    centroids = pd.DataFrame(kmeans.cluster_centers_)
    print("Centroids")
    print(centroids)
    print()

    print("Cluster sizes")
    sizes = cluster_sizes(clusters, data)
    for c in sizes.keys():
        print(f"Cluster", c, " = ", sizes[c])

    print()

    #######################################################

    spread = (centroids.max() - centroids.min()) / data.std()
    display(spread.sort_values(ascending=False).to_frame("centroid spread (in std devs)"))
    print()

    #######################################################

    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data))
    data_scaled.describe()

    #######################################################

    if DO_RANGE:
        wcss = []
        for i in range(1, 11):
            print(f"Testing K={i}")
            km = KMeans(n_clusters=i,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        random_state=0)
            km.fit(data_scaled)
            wcss.append(km.inertia_)

        plt.plot(range(1, 11), wcss, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.tight_layout()
        # pl.savefig('elbow-kmeans.png', dpi=300)
        plt.show()


def main() -> int:
    run()

    return 0

if __name__ == "__main__":
    sys.exit(main())
