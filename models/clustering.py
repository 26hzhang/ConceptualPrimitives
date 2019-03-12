import numpy as np
from sklearn.preprocessing import normalize
from collections import OrderedDict
from sklearn import cluster, metrics
from scipy.spatial.distance import cosine, euclidean
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
    for key, value in results.items():
        value = dict(value)
        value = OrderedDict(sorted(value.items(), key=lambda kv: kv[1]))
        if not keep_score:
            value = list(value)
        cluster_dict[key] = value
    cluster_dict = OrderedDict(sorted(cluster_dict.items(), key=lambda kv: kv[0]))
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
