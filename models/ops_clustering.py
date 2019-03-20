import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn import cluster, metrics
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine, euclidean
from utils.data_utils import write_pickle, load_pickle
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def normalize_vectors(vectors, norm_method="l2"):
    return normalize(vectors, axis=1, norm=norm_method)


def kmeans_clustering(vectors, clusters, init="k-means++", n_init=20, max_iter=500, tol=1e-12, verbose=0):
    kmeans = cluster.KMeans(n_clusters=clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose,
                            precompute_distances="auto", random_state=None, n_jobs=10, algorithm="auto")
    kmeans.fit(vectors)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score = kmeans.score(vectors)
    silhouette_score = metrics.silhouette_score(vectors, labels, metric="cosine")
    return labels, centroids, score, silhouette_score


def compute_distance(vocab, labels, vectors, centroids, dist_method="cosine", keep_score=False):
    results = dict()
    for verb, label in zip(vocab, labels):
        if dist_method == "cosine":
            distance = cosine(centroids[label], vectors[vocab.index(verb)])
        else:
            distance = euclidean(centroids[label], vectors[vocab.index(verb)])
        results[label] = results.get(label, []) + [(verb, distance)]
    cluster_dict = dict()
    cluster_score_dict = dict()
    for key, value in results.items():
        value = sorted(value, key=lambda kv: kv[1])
        words, scores = list(), list()
        for word, score in value:
            words.append(word)
            scores.append(score)
        value = dict(value)
        cluster_dict[key] = value
        cluster_score_dict[key] = scores
    cluster_dict = OrderedDict(sorted(cluster_dict.items(), key=lambda kv: kv[0]))
    cluster_score_dict = OrderedDict(sorted(cluster_score_dict.items(), key=lambda kv: kv[0]))
    if keep_score:
        return cluster_dict, cluster_score_dict
    else:
        return cluster_dict


def compute_knearest(verb, vocab, vectors, dist_method="cosine", top_k=100):
    top_knearest = dict()
    verb_idx = vocab.index(verb)
    temp_list = [verb_idx]
    while True:
        verb_indices = temp_list.copy()
        temp_list = []
        for v_idx in verb_indices:
            dist_dict = dict()
            verb_vector = np.reshape(vectors[v_idx], newshape=(1, vectors.shape[1]))
            if dist_method == "cosine":
                distances = cosine_distances(verb_vector, vectors)
            else:
                distances = euclidean_distances(verb_vector, vectors)
            distances = list(np.reshape(distances, newshape=(distances.shape[1],)))
            for idx, distance in enumerate(distances):
                if idx == v_idx:
                    continue
                dist_dict[idx] = distance
            dist_dict = OrderedDict(sorted(dist_dict.items(), key=lambda kv: kv[1]))
            top_5 = list(dist_dict.items())[0:5]
            for key, value in top_5:
                if key not in top_knearest and key != verb_idx:
                    top_knearest[key] = value
                    temp_list.append(key)
        if len(top_knearest) >= top_k:
            break
    top_knearest = OrderedDict(sorted(top_knearest.items(), key=lambda kv: kv[1]))
    top_knearest = [vocab[idx] for idx in list(top_knearest)[0:top_k]]
    return top_knearest


def clustering(vectors, vocab, num_clusters, cluster_method="kmeans", save_path=None, norm=True, norm_method="l2"):
    if save_path is not None and os.path.exists(save_path):
        return load_pickle(save_path)
    else:
        if norm:
            vectors = normalize_vectors(vectors, norm_method=norm_method)

        print("k-means clustering...")
        labels, centroids, score, silhouette_score = kmeans_clustering(vectors,
                                                                       clusters=num_clusters,
                                                                       init="k-means++",
                                                                       n_init=20,
                                                                       max_iter=10000,
                                                                       tol=1e-12,
                                                                       verbose=0)
        print("Score (opposite of the value of embeddings on the K-means objective) is the sum of {}".format(score))
        print("Silhouette score: {}".format(silhouette_score))

        clusters = compute_distance(vocab=vocab,
                                    labels=labels,
                                    vectors=vectors,
                                    centroids=centroids,
                                    dist_method="cosine",
                                    keep_score=False)

        if cluster_method == "kmeans":
            write_pickle(clusters, filename=save_path)
            return clusters

        elif cluster_method == "knearest":
            clusters_dict = dict()
            for cluster_idx, verb in tqdm(clusters.items(), total=len(clusters), desc="compute k-nearest verbs"):
                # key_verb = next(iter(verb))
                key_verb = verb[0]
                sub_verbs = compute_knearest(verb=key_verb,
                                             vocab=vocab,
                                             vectors=vectors,
                                             dist_method="cosine",
                                             top_k=100)
                clusters_dict[cluster_idx] = [key_verb] + sub_verbs
            write_pickle(clusters_dict, filename=save_path)
            return clusters_dict

        else:
            raise ValueError("Unsupported clustering method, only [kmeans | knearest] are utilized!!!")
