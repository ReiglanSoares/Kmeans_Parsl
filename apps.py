from parsl import python_app
import numpy as np

@python_app
def kmeans_fragment(fragment_id,
                    points_per_fragment,
                    dimensions,
                    centroids):
    
    np.random.seed(fragment_id)

    points = np.random.random(
        (points_per_fragment, dimensions)
    ).astype(np.float64)

    K = centroids.shape[0]

    partial_sum = np.zeros((K, dimensions), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)

    # Distância ponto → todos os centróides
    for p in points:
        dists = np.linalg.norm(centroids - p, axis=1)
        cid = np.argmin(dists)
        partial_sum[cid] += p
        counts[cid] += 1

    return partial_sum, counts

@python_app
def reduce_and_update(old_centroids, *partials):
    K, D = old_centroids.shape

    total_sum = np.zeros((K, D), dtype=np.float64)
    total_count = np.zeros(K, dtype=np.int64)

    for psum, cnt in partials:
        total_sum += psum
        total_count += cnt

    new_centroids = old_centroids.copy()

    for k in range(K):
        if total_count[k] > 0:
            new_centroids[k] = total_sum[k] / total_count[k]

    return new_centroids                      
