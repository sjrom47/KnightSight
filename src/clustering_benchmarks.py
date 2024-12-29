import cv2
import numpy as np
from utils import show_image
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import time
from scipy.cluster.hierarchy import dendrogram, linkage


def label_to_color_image(label_img, mean_colors):
    color_img = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)

    for label, mean_color in mean_colors.items():
        color_img[label_img == label] = mean_color

    return color_img  # Output is in BGR format


def compute_mean_colors(image, labels):
    unique_labels = np.unique(labels)
    mean_colors = {}
    for label in unique_labels:
        mask = labels == label
        mean_color = np.mean(image[mask], axis=0)
        mean_colors[label] = mean_color
    return mean_colors


if __name__ == "__main__":
    # Load the image
    image = cv2.imread("data/labeled_data/knight/knight_001.jpg")
    # image = cv2.imread("data/other_data/pieces/bishop/bishop_001.jpg")

    show_image(image)

    # Resize the image for clustering
    small_image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
    # small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2HSV)
    reshaped_image = small_image.reshape(-1, 3)

    # Benchmark DBSCAN
    start_time = time.time()
    dbscan = DBSCAN(eps=2.5, min_samples=10, n_jobs=-1).fit(reshaped_image)
    dbscan_labels = dbscan.labels_.reshape(small_image.shape[:2])
    dbscan_mean_colors = compute_mean_colors(small_image, dbscan_labels)
    dbscan_time = time.time() - start_time
    print(f"DBSCAN Time: {dbscan_time:.4f} seconds")

    # Benchmark Agglomerative Clustering
    start_time = time.time()
    agglomerative = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(
        reshaped_image
    )
    agglomerative_labels = agglomerative.labels_.reshape(small_image.shape[:2])
    agglomerative_mean_colors = compute_mean_colors(small_image, agglomerative_labels)
    agglomerative_time = time.time() - start_time
    print(f"Agglomerative Clustering Time: {agglomerative_time:.4f} seconds")

    # Benchmark Gaussian Mixture Models
    start_time = time.time()
    gmm = GaussianMixture(n_components=5, covariance_type="full").fit(reshaped_image)
    gmm_labels = gmm.predict(reshaped_image).reshape(small_image.shape[:2])
    gmm_mean_colors = compute_mean_colors(small_image, gmm_labels)
    gmm_time = time.time() - start_time
    print(f"GMM Time: {gmm_time:.4f} seconds")

    # Benchmark KMeans
    start_time = time.time()
    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10).fit(reshaped_image)
    kmeans_labels = kmeans.predict(reshaped_image).reshape(small_image.shape[:2])
    kmeans_mean_colors = compute_mean_colors(small_image, kmeans_labels)
    kmeans_time = time.time() - start_time
    print(f"KMeans Time: {kmeans_time:.4f} seconds")

    # Display results
    dbscan_img = label_to_color_image(dbscan_labels, dbscan_mean_colors)
    show_image(dbscan_img)
    agglomerative_img = label_to_color_image(
        agglomerative_labels, agglomerative_mean_colors
    )
    show_image(agglomerative_img)
    gmm_img = label_to_color_image(gmm_labels, gmm_mean_colors)
    show_image(gmm_img)
    kmeans_img = label_to_color_image(kmeans_labels, kmeans_mean_colors)
    show_image(kmeans_img)
