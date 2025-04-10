import numpy as np
import matplotlib.pyplot as plt

# Produces time-series data
def prod_db(no_of_points, no_of_features):
    db = np.random.randn(no_of_points, no_of_features)
    return db

# Modified k-means algorithm
def algo_mod_k_m(db, max_k=10):
    # centroids are initialised with random values
    centrd = np.random.rand(6, db.shape[1])
    prev_centrd = centrd.copy()

    while True:
        # Points will be put into the nearest centroid of cluster
        labl = np.argmin(np.linalg.norm(db[:, np.newaxis] - centrd, axis=2), axis=1)

        # Centroid will be updated on basis of cluster mean
        for i in range(centrd.shape[0]):
            pts_in_clus = db[labl == i]
            if len(pts_in_clus) > 0:
                centrd[i] = np.mean(pts_in_clus, axis=0)

        # Convergence will be checked
        if np.allclose(prev_centrd, centrd):
            break
        prev_centrd = centrd.copy()

    return labl, centrd

# Calculating within cluster sum of squares
def calc_wcss(db, centrd, labl):
    wc_ss = 0
    for i in range(centrd.shape[0]):
        cluster_points = db[labl == i]
        wc_ss += np.sum(np.linalg.norm(cluster_points - centrd[i], axis=1) ** 2)
    return wc_ss

# Plotting elbow method
def elb_met_p(db, max_k=10):
    wcss_values = []
    for k in range(1, max_k + 1):
        labl, centrd = algo_mod_k_m(db, max_k=k)
        wcss = calc_wcss(db, centrd, labl)
        wcss_values.append(wcss)
    plt.plot(range(1, max_k + 1), wcss_values, marker='o')
    plt.xlabel('Num of clusters (k)')
    plt.ylabel('WCSS')
    plt.title('The Elbow Method')
    plt.show()

    # k will be chosen
    opti_k = np.argmin(np.diff(wcss_values)) + 1
    print("The number of clusters formed are :", opti_k)

    # Clusters are formed based on k
    labl, centrd = algo_mod_k_m(db, max_k=opti_k)
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(opti_k):
        cluster_points = db[labl == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}', color=colors[i % len(colors)])
    plt.scatter(centrd[:, 0], centrd[:, 1], marker='x', color='black', label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Clusters Formation')
    plt.legend()
    plt.show()

# Producing time-series data
db = prod_db(100, 2)

# Plotting elbow method and respective clusters
elb_met_p(db)

