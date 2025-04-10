import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Loading the dataset
db = np.loadtxt('synthetic_control.data')

# Ground truth is assumed to be 10
n_gr_tr = 10

# labels of ground truth to be produced on the basis of number of clusters
gr_tr = np.repeat(np.arange(n_gr_tr), len(db) // n_gr_tr)

# Modified k-means algorithm
def algo_mod_k_m(db, num_clusters):
    k_me = KMeans(n_clusters=num_clusters, init='random')
    clus_l = k_me.fit_predict(db)
    return clus_l

# Different number of clusters to be formed and choose the one with most score
h_ari_sc = -1
h_nmi_sc = -1
h_no_clus = -1

# Setting the range from 4 to 26
for num_clusters in range(4, 27):
    clus_lbl = algo_mod_k_m(db, num_clusters)
    ari_sc = adjusted_rand_score(gr_tr, clus_lbl)
    nmi_sc= normalized_mutual_info_score(gr_tr, clus_lbl)

# In case of better score then update it
    if ari_sc > h_ari_sc:
        h_ari_sc = ari_sc
        h_no_clus = num_clusters
    if nmi_sc > h_nmi_sc:
        h_nmi_sc = nmi_sc

# Implement k-means cluster
best_cluster_labels = algo_mod_k_m(db, h_no_clus)

# Printing
print(f'Best Adjusted Rand Index (ARI): {h_ari_sc} for {h_no_clus} clusters')
print(f'Best Normalized Mutual Information (NMI): {h_nmi_sc}')
