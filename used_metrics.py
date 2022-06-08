"""
Master Thesis
Academic year 2021-2022

Authors:
    - GILSON Maxence
"""

###########
# Imports #
###########

import os
import statistics

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score  # Davies-Bouldin Index
from sklearn.metrics import silhouette_samples, silhouette_score
import unsupervised_methods
from Unsup_Feature_Selection import feature_selection


def get_name_metric(metric):
    if metric == 'silhouette':
        return 'Silhouette score'
    if metric == 'DB':
        return 'Davies-Bouldin index'
    if metric == 'CH':
        return 'Calinski-Harabasz Index'
    if metric == 'Hamming':
        return 'Hamming variance score'
    else:
        return 'Err'


# getting a specific score of a metric for a specific clustering method
def get_score(metric, DB, method, n_cluster, max_iter, n_init):
    if metric == 'CH':
        clusters = unsupervised_methods.methods(DB, method, n_cluster, max_iter, n_init, True)
        if method == 1:
            DB = DB.drop('INDUSTRY', inplace=False, axis=1)
        score = metrics.calinski_harabasz_score(DB, clusters)
        print('Calinski - Harabasz score is ', score)
        return
    elif metric == 'silhouette':
        clusters = unsupervised_methods.methods(DB, method, n_cluster, max_iter, n_init, True)
        if method == 1:
            DB = DB.drop('INDUSTRY', inplace=False, axis=1)
        score = silhouette_score(DB, clusters)
        print('Silhouette score is', score)
        return
    elif metric == 'DB':
        clusters = unsupervised_methods.methods(DB, method, n_cluster, max_iter, n_init, True)
        if method == 1:
            DB = DB.drop('INDUSTRY', inplace=False, axis=1)
        score = davies_bouldin_score(DB, clusters)
        print('Davies-Bouldin score is ', score)
        return
    elif metric == 'Hamming':
        clusters = unsupervised_methods.methods(DB, method, n_cluster, max_iter, n_init, False)
        if method == 1:
            DB = DB.drop('INDUSTRY', inplace=False, axis=1)
        score2 = hamming_distance(DB, method, clusters, n_cluster)[1]
        print('Hamming variance score is ', score2)
        return
    else:
        return 'Err : Wrong metric'


# Creating new directory to save the figures
def get_directory(method):
    name_dir = ""
    script_dir = os.path.dirname(__file__)
    if method == 1:
        name_dir = "Results/k_modes_scores/"
    if method == 2:
        name_dir = "Results/k_means_scores/"
    if method == 3:
        name_dir = "Results/bi_k_means_scores/"
    if method == 4:
        name_dir = "Results/SOM_scores/"
    if method == 5:
        name_dir = "Results/Boosting_scores/"
    if method == "RF":
        name_dir = "Results/RF/"
    if method == "SVM":
        name_dir = "Results/SVM/"
    if method == "corr":
        name_dir = "Results/Correlation_maps/"
    if not method:
        name_dir = "Results/Tests/"

    results_dir = os.path.join(script_dir, name_dir)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    return results_dir


# Loading the row to compute the hamming distance between rows
def load_row_bdd(n_row, bdd, method):
    if method == 1:
        size_bdd = bdd.shape
    else:
        size_bdd = bdd.shape
    M = []
    for i in range(size_bdd[1]):
        M.append(bdd.iat[n_row, i])
    return M


# Computing hamming intra-cluster distance #
############################################

def hamming_distance(bdd, method, cluster, k_cluster):
    if method == 1:
        size_bdd = bdd.shape
    else:
        size_bdd = bdd.shape
    Intra = []
    Inter = []
    hamming_value = [0, 0]
    i = 0
    if method == 4:
        labels = cluster
    else:
        labels = cluster.labels_
    for k in range(k_cluster):
        for x in range(size_bdd[0]):
            if labels[x] == k:
                M = load_row_bdd(x, bdd, method)
                i = x + 1
                for j in range(i, size_bdd[0]):
                    if labels[j] == k:
                        N = load_row_bdd(j, bdd, method)
                        Intra.append(distance.hamming(M, N))
                    if labels[j] != k:
                        N = load_row_bdd(j, bdd, method)
                        Inter.append(distance.hamming(M, N))
    hamming_value[1] = statistics.variance(Inter) / statistics.variance(Intra)
    return hamming_value


# inertia_ Sum of squared distances of samples to their
# closest cluster center, weighted by the sample weights if provided
def plot_elbow_method(bdd, metric, method, n_clusters_max, max_iter, n_init, direct, NAME, FS):
    # Deleting the first column the database if the method
    # is K-modes. Metrics used don't accept categorical data
    M = []
    N = []
    K = range(2, n_clusters_max + 1)
    directory = get_directory(direct)
    if method == 1:
        if not FS:
            Mod_bdd = bdd.drop('INDUSTRY', inplace=False, axis=1)
        else:
            Mod_bdd = bdd
        for k in K:
            if metric == 'CH':
                clusters = unsupervised_methods.methods(bdd, method, k, max_iter, n_init, True)
                M.append(metrics.calinski_harabasz_score(Mod_bdd, clusters))
            elif metric == 'silhouette':
                clusters = unsupervised_methods.methods(bdd, method, k, max_iter, n_init, True)
                M.append(silhouette_score(Mod_bdd, clusters))
            elif metric == 'DB':
                clusters = unsupervised_methods.methods(bdd, method, k, max_iter, n_init, True)
                M.append(davies_bouldin_score(Mod_bdd, clusters))
            elif metric == 'Hamming':
                clusters = unsupervised_methods.methods(bdd, method, k, max_iter, n_init, False)
                M.append(hamming_distance(Mod_bdd, method, clusters, k)[1])
            else:
                return 'Err : Wrong metric'

    if method != 1:
        for k in K:
            if metric == 'distortion':
                clusters = unsupervised_methods.methods(bdd, method, k, max_iter, n_init, False)
                M.append(clusters.inertia_)
            elif metric == 'silhouette':
                clusters = unsupervised_methods.methods(bdd, method, k, max_iter, n_init, True)
                M.append(silhouette_score(bdd, clusters))
            elif metric == 'CH':
                clusters = unsupervised_methods.methods(bdd, method, k, max_iter, n_init, True)
                M.append(metrics.calinski_harabasz_score(bdd, clusters))
            elif metric == 'DB':
                clusters = unsupervised_methods.methods(bdd, method, k, max_iter, n_init, True)
                M.append(davies_bouldin_score(bdd, clusters))
            elif metric == 'Hamming':
                if method == 4:
                    clusters = unsupervised_methods.methods(bdd, method, k, max_iter, n_init, True)
                else:
                    clusters = unsupervised_methods.methods(bdd, method, k, max_iter, n_init, False)
                M.append(hamming_distance(bdd, method, clusters, k)[1])
            else:
                return 'Err : Wrong metric'
    print(metric)
    print(M)
    plt.figure(figsize=(12, 8))
    if FS:
        return M
    plt.plot(K, M, 'bx-')
    plt.xlabel('k clusters')
    if not NAME:
        name = unsupervised_methods.get_name_method(method)
    else:
        name = NAME
    name += "_Elbow_"
    if metric == 'silhouette':
        plt.ylabel('Silhouette score')
        # plt.title('The Elbow Method showing the optimal k')
        name += metric
        plt.savefig(directory + name)
    if metric == 'Hamming':
        plt.ylabel('Hamming variance')
        # plt.title('The Elbow Method showing the optimal k')
        name += metric
        plt.savefig(directory + name)
    elif metric == 'distortion':
        plt.ylabel('Distortion')
        # plt.title('The Elbow Method showing the optimal k')
        name += metric
        plt.savefig(directory + name)
    elif metric == 'CH':
        plt.ylabel('Calinski-Harabasz Index')
        # plt.title('The Elbow Method showing the optimal k')
        name += metric
        plt.savefig(directory + name)
    elif metric == 'DB':
        plt.ylabel('Davies-Bouldin Index')
        # plt.title('The Elbow Method showing the optimal k')
        name += metric
        plt.savefig(directory + name)
    plt.close()


# Inspired from scikit-learn "Selecting the number of clusters with silhouette analysis on KMeans clustering"
def plot_silhouettte_score(bdd, method, n_clusters_max, max_iter, n_init, NAME):
    K = range(2, n_clusters_max + 1)
    M = []
    range_name = ["2", "3", "4", "5", "6", "7", "8", "9", "10"]
    if not NAME:
        directory = get_directory(method)
    else:
        directory = get_directory(False)
    j = 0
    if method == 1:
        bdd = bdd.drop('INDUSTRY', inplace=False, axis=1)
    for k in K:
        plt.figure(figsize=(16, 8))

        # The 1st subplot is the silhouette plot
        plt.xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        plt.ylim([0, len(bdd) + (k + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        cluster_labels = unsupervised_methods.methods(bdd, method, k, max_iter, n_init, True)

        # The silhouette_score gives the average value for all the samples.
        silhouette_avg = silhouette_score(bdd, cluster_labels)
        M.append(silhouette_avg)
        print(
            "For n_clusters =",
            k,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(bdd, cluster_labels)

        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / k)
            plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        plt.title("The silhouette plot for %d clusters " % k)
        plt.xlabel("The silhouette coefficient values")
        plt.ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")

        # Clear the yaxis labels
        plt.yticks([])  # Clear the yaxis labels
        plt.xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        name_method = unsupervised_methods.get_name_method(method)
        name = range_name[j] + name_method
        if NAME:
            name += NAME

        j += 1

        plt.savefig(directory + name)
        plt.close()
    plt.figure(figsize=(12, 8))
    plt.plot(K, M, 'bx-')
    plt.xlabel('k clusters')
    name += "Plot_avg_silhouette"
    plt.ylabel('Average Silhouette Score')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig(directory + name)


# Plot an elbow plot comparing the same clustering methods using two different datasets and the same metric
def plot_comparison_elbow(DB, metric, method, n_clusters_max, max_iter, n_init, NAME):
    M = plot_elbow_method(bdd=DB, metric=metric, method=method, n_clusters_max=n_clusters_max, max_iter=max_iter,
                          n_init=n_init, direct=False, NAME=NAME, FS=True)
    if metric == 'silhouette':
        if method == 1:
            N = [0.134196, 0.152390, 0.146460, 0.184567, 0.195509, 0.185716, 0.202877, 0.198732, 0.186096]
        elif method == 2:
            N = [0.154452, 0.162242, 0.173736, 0.193562, 0.214856, 0.220040, 0.216222, 0.205982, 0.202425]
        elif method == 3:
            N = [0.154452, 0.153107, 0.171411, 0.188285, 0.210528, 0.186118, 0.184816, 0.170716, 0.175047]
        else:
            return 'Err'
    if metric == 'DB':
        if method == 1:
            N = [2.210586, 1.866099, 2.106203, 1.819836, 1.777216, 1.762776, 1.786505, 1.718527, 1.640137]
        elif method == 2:
            N = [2.213624, 2.001298, 2.047795, 1.825063, 1.784464, 1.650407, 1.694081, 1.688460, 1.571785]
        elif method == 3:
            N = [2.213624, 2.273729, 2.013843, 1.880885, 1.696555, 1.828724, 1.675803, 1.672129, 1.604943]
        else:
            return 'Err'
    if metric == 'CH':
        if method == 1:
            N = [15.778262, 14.687782, 13.132777, 12.406879, 13.429965, 12.133495, 11.623234, 10.679703, 9.963981]
        elif method == 2:
            N = [17.427595, 15.390047, 14.262858, 13.705986, 13.371051, 12.491318, 11.852183, 11.179833, 10.636045]
        elif method == 3:
            N = [17.427595, 13.605782, 14.004665, 12.957827, 13.012042, 12.081386, 11.016905, 10.180744, 9.717669]
        else:
            return 'Err'
    if metric == 'Hamming':
        if method == 1:
            N = [0.782028, 0.856504, 0.728932, 0.649732, 1.084786, 0.990638, 1.085631, 1.348115, 1.303051]
        elif method == 2:
            N = [0.575848, 0.546311, 0.609325, 0.665921, 0.932545, 0.908713, 0.998828, 1.124170, 1.150373]
        elif method == 3:
            N = [0.575848, 0.630642, 0.633811, 0.741963, 0.972333, 0.974422, 1.083660, 1.119070, 1.208959]
        else:
            return 'Err'
    K = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    directory = get_directory(False)
    plt.figure(figsize=(12, 8))
    plt.plot(K, M, 'bx-')
    plt.plot(K, N, color='green', marker='o')
    plt.xlabel('k clusters')
    plt.ylabel(get_name_metric(metric))
    plt.savefig(directory + NAME)
    plt.close()


# Compare the elbow plot of the different clutering methods before and after feature selection
def comparing_metrics_FS(old_DB, new_DB, metric, n_clusters_max, max_iter, n_init, NAME):
    global N_bi
    # computing the values of the metrics for the different clustering methods after feature selection
    M_kmodes = plot_elbow_method(bdd=new_DB, metric=metric, method=1, n_clusters_max=n_clusters_max, max_iter=max_iter,
                                 n_init=n_init, direct=False, NAME=NAME, FS=True)
    N_kmodes = plot_elbow_method(bdd=old_DB, metric=metric, method=1, n_clusters_max=n_clusters_max, max_iter=max_iter,
                                 n_init=n_init, direct=False, NAME=NAME, FS=True)
    M_kmeans = plot_elbow_method(bdd=new_DB, metric=metric, method=2, n_clusters_max=n_clusters_max, max_iter=max_iter,
                                 n_init=n_init, direct=False, NAME=NAME, FS=True)
    N_kmeans = plot_elbow_method(bdd=old_DB, metric=metric, method=2, n_clusters_max=n_clusters_max, max_iter=max_iter,
                                 n_init=n_init, direct=False, NAME=NAME, FS=True)
    M_bi = plot_elbow_method(bdd=new_DB, metric=metric, method=3, n_clusters_max=n_clusters_max, max_iter=max_iter,
                             n_init=n_init, direct=False, NAME=NAME, FS=True)
    N_bi = plot_elbow_method(bdd=old_DB, metric=metric, method=3, n_clusters_max=n_clusters_max, max_iter=max_iter,
                             n_init=n_init, direct=False, NAME=NAME, FS=True)
    directory = get_directory(False)
    K = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # x markers = clustering methods after feature selection
    # o markers = clustering methods before feature selection
    # red color = k-modes algorithm
    # green color = k-means algorithm
    # orange color = bisecting k-means algorithm
    plt.figure(figsize=(12, 8))
    plt.plot(K, M_kmodes, 'rx-')
    plt.plot(K, N_kmodes, color='red', marker='o')
    plt.plot(K, M_kmeans, 'gx-')
    plt.plot(K, N_kmeans, color='green', marker='o')
    plt.plot(K, M_bi, color='orange', marker='x')
    plt.plot(K, N_bi, color='orange', marker='o')
    plt.xlabel('k clusters')
    plt.ylabel(get_name_metric(metric))
    plt.savefig(directory + NAME)
    plt.close()


# plotting all values for k-means after the different feature selection methods
def comparing_ALL_FS(DB, metric, max_iter, n_init, NAME):
    FS_techniques = ['MCFS', 'Lap Score', 'SPEC', 'USFSM', 'Low Variance']
    new_bdd = DB[feature_selection(FS_techniques[1], 6)]
    lapscor = plot_elbow_method(bdd=new_bdd, metric=metric, method=2, n_clusters_max=10, max_iter=max_iter,
                                n_init=n_init, direct=False, NAME=NAME, FS=True)
    new_bdd = DB[feature_selection(FS_techniques[2], 6)]
    spec = plot_elbow_method(bdd=new_bdd, metric=metric, method=2, n_clusters_max=10, max_iter=max_iter,
                             n_init=n_init, direct=False, NAME=NAME, FS=True)
    new_bdd = DB[feature_selection(FS_techniques[3], 6)]
    usfsm = plot_elbow_method(bdd=new_bdd, metric=metric, method=2, n_clusters_max=10, max_iter=max_iter,
                              n_init=n_init, direct=False, NAME=NAME, FS=True)
    new_bdd = DB[feature_selection(FS_techniques[0], 6)]
    mcfs = plot_elbow_method(bdd=new_bdd, metric=metric, method=2, n_clusters_max=10, max_iter=max_iter,
                             n_init=n_init, direct=False, NAME=NAME, FS=True)
    new_bdd = DB[feature_selection(FS_techniques[4], 6)]
    lowvar = plot_elbow_method(bdd=new_bdd, metric=metric, method=2, n_clusters_max=10, max_iter=max_iter,
                               n_init=n_init, direct=False, NAME=NAME, FS=True)
    directory = get_directory(False)
    K = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.figure(figsize=(12, 8))
    plt.plot(K, lapscor, color='purple')
    plt.plot(K, spec, color='red')
    plt.plot(K, usfsm, color='aqua')
    plt.plot(K, mcfs, color='green')
    plt.plot(K, lowvar, color='orange')
    plt.xlabel('k clusters')
    plt.ylabel(get_name_metric(metric))
    plt.savefig(directory + NAME)
    plt.close()
