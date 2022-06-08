"""
Master Thesis
Academic year 2021-2022

Authors:
    - GILSON Maxence
"""

###########
# Imports #
###########
import pandas as pd

from supervised_methods import sup_method
from unsupervised_methods import methods
from sklearn.metrics import silhouette_samples, silhouette_score
from Unsup_Feature_Selection import feature_selection
from database import DB_np, DB, DB_Cat
from used_metrics import plot_elbow_method, plot_silhouettte_score, get_score, plot_comparison_elbow \
    , comparing_metrics_FS, comparing_ALL_FS


def compare_methods(metric, method, n_clusters_max, max_iter, n_init, name_1, name_2, FS_technique, n_features, elbow):
    # selection = feature_selection(FS_technique, n_features)
    new_DB = get_new_DB(DB, feature_selection(FS_technique, n_features))
    if elbow:
        plot_elbow_method(DB, metric, method, n_clusters_max, max_iter, n_init, False, name_1, False)
        plot_elbow_method(new_DB, metric, method, n_clusters_max, max_iter, n_init, False, name_2, False)
    else:
        plot_silhouettte_score(DB, method, n_clusters_max, max_iter, n_init, name_1)
        plot_silhouettte_score(new_DB, method, n_clusters_max, max_iter, n_init, name_2)
    return


def compare_FS(n_features):
    FS_techniques = ['MCFS', 'Lap Score', 'SPEC', 'USFSM', 'Low Variance']
    for i in range(0, 5):
        print(feature_selection(FS_techniques[i], n_features))
    return


def get_new_DB(db, name_features):
    return db[name_features]


if __name__ == '__main__':
    # method : int
    # The number of the method used
    # 	- 1 = k-modes
    # 	- 2 = k-means
    # 	- 3 = bisecting k-means
    #   - 4 = SOM

    method = 2
    n_clusters = 6
    max_iter = 300
    n_init = 100
    verbose = 0
    predict = True
    n_clusters_max = 10
    metric = ['silhouette', 'Hamming', 'CH', 'DB']
    FS_techniques = ['MCFS', 'Lap Score', 'SPEC', 'USFSM', 'Low Variance']
    name_1 = 'old_DB'
    name_2 = 'new_DB'

    # Comparing the features selected by the diff feature selection methods
    # compare_FS(n_features=6)

    # Plotting all plots for kmodes
    # for i in range(0, 4):
    #     plot_elbow_method(DB_Cat, metric[i], 1, n_clusters_max, max_iter, n_init, 1, False, False)
    # plot_silhouettte_score(DB_Cat, 1, n_clusters_max, max_iter, n_init,False)

    # Plotting all plots for kmeans
    # for i in range(0, 4):
    #     plot_elbow_method(DB, metric[i], 2, n_clusters_max, max_iter, n_init, 2, False, False)
    # plot_silhouettte_score(DB, 2, n_clusters_max, max_iter, n_init,False)

    # Plotting all plots for bisecting kmeans
    # for i in range(0, 4):
    #     plot_elbow_method(DB, metric[i], 3, n_clusters_max, max_iter, n_init, 3, False, False)
    # plot_silhouettte_score(DB, 3, n_clusters_max, max_iter, n_init,False)

    # Plotting all plots for SOM
    # for i in range(0, 4):
    #     plot_elbow_method(DB, metric[i], 4, n_clusters_max, max_iter, n_init, 4, False, False)
    # plot_silhouettte_score(DB, 4, n_clusters_max, max_iter, n_init, False)

    # Plotting 2 graphs comparing one metric of one clustering method but before and after feature selection
    # compare_methods(metric='CH', method=2, n_clusters_max=n_clusters_max, max_iter=max_iter, n_init=n_init,
    #                 name_1=name_1, name_2=name_2, FS_technique='Low Variance', n_features=10, elbow=True)

    # Get names of features for every feature selection method
    # for i in range(0, 5):
    #     print(feature_selection(FS_techniques[i], 6))

    # Create the plots comparing all metrics of all clustering methods for one feature selection method and create
    # a silhouette representation of the created clusters
    # Name = ['_silhouette', '_Hamming', '_CH', '_DB']
    # for j in range(0, 5):
    #     new_bdd = get_new_DB(DB, feature_selection(FS_techniques[j], 6))
    #     plot_silhouettte_score(new_bdd, 2, n_clusters_max, max_iter, n_init, False)
    #     for i in range(0, 4):
    #         comparing_metrics_FS(DB, new_bdd, metric[i], n_clusters_max, max_iter, n_init, FS_techniques[j]+Name[i])

    # Compare all feature selection techniques on K-Means for a specific metric
    # comparing_ALL_FS(DB, 'CH', max_iter, n_init, 'CH_FS')
    # comparing_ALL_FS(DB, 'DB', max_iter, n_init, 'DB_FS')
    # comparing_ALL_FS(DB, 'silhouette', max_iter, n_init, 'silhouette_FS')
    # comparing_ALL_FS(DB, 'Hamming', max_iter, n_init, 'Hamming_FS')

    # Give the final cluster labels for our dataset
    # New_DB = get_new_DB(DB, feature_selection('SPEC', 6))
    # Labels = methods(New_DB, method=1, n_clusters=5, max_iter=max_iter, n_init=n_init, predict=True)
    # print(Labels)
