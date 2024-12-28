import cv2
import numpy as np
from utils import show_image, load_images
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt


def label_to_color_image(label_img, mean_colors):
    color_img = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)

    for label, mean_color in mean_colors.items():
        color_img[label_img == label] = mean_color

    return color_img


def compute_mean_colors(image, labels):
    unique_labels = np.unique(labels)
    mean_colors = []
    for label in unique_labels:
        mask = labels == label
        mean_color = np.mean(image[mask], axis=0)
        mean_colors.append(mean_color)
    return mean_colors


if __name__ == "__main__":
    # Load the image
    # imgs = load_images("data/labeled_data/pawn/*.jpg")
    imgs = load_images("data/other_data/pieces/*/*.jpg")
    centroid_list = []
    blue_centroid_list = []
    yellow_centroid_list = []
    real_colour_values = []
    mean_rgb_values = []
    yellow_mean_rgb_values = []
    blue_mean_rgb_values = []
    for img in imgs:
        small_image = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
        mean_rgb_values.append(np.mean(small_image, axis=(0, 1)))
        reshaped_image = small_image.reshape(-1, 3)
        agglomerative = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(
            reshaped_image
        )
        agglomerative_labels = agglomerative.labels_.reshape(small_image.shape[:2])
        agglomerative_mean_colors = compute_mean_colors(
            small_image, agglomerative_labels
        )
        cv2.imshow("image", img)
        key = cv2.waitKey(0)
        if key == ord("b"):
            blue_centroid_list.extend(agglomerative_mean_colors)
            blue_mean_rgb_values.append(np.mean(small_image, axis=(0, 1)))
        elif key == ord("y"):
            yellow_centroid_list.extend(agglomerative_mean_colors)
            yellow_mean_rgb_values.append(np.mean(small_image, axis=(0, 1)))
        centroid_list.extend(agglomerative_mean_colors)
        cv2.destroyAllWindows()
    centroid_list = np.array(centroid_list)
    blue_centroid_list = np.array(blue_centroid_list)
    yellow_centroid_list = np.array(yellow_centroid_list)
    centroid_list = cv2.cvtColor(
        centroid_list.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)
    blue_centroid_list = cv2.cvtColor(
        blue_centroid_list.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)
    yellow_centroid_list = cv2.cvtColor(
        yellow_centroid_list.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        centroid_list[:, 0],
        centroid_list[:, 1],
        centroid_list[:, 2],
        color=centroid_list / 255.0,
    )
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    # plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        blue_centroid_list[:, 0],
        blue_centroid_list[:, 1],
        blue_centroid_list[:, 2],
        color="blue",
        label="Blue",
    )
    ax.scatter(
        yellow_centroid_list[:, 0],
        yellow_centroid_list[:, 1],
        yellow_centroid_list[:, 2],
        color="yellow",
        label="Yellow",
    )
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    plt.legend()
    plt.show()
    mean_rgb_values = np.array(mean_rgb_values)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        mean_rgb_values[:, 0],
        mean_rgb_values[:, 1],
        mean_rgb_values[:, 2],
        color=mean_rgb_values / 255.0,
    )
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    plt.legend()
    plt.show()
    yellow_mean_rgb_values = np.array(yellow_mean_rgb_values)
    blue_mean_rgb_values = np.array(blue_mean_rgb_values)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        blue_mean_rgb_values[:, 0],
        blue_mean_rgb_values[:, 1],
        blue_mean_rgb_values[:, 2],
        color="blue",
    )
    ax.scatter(
        yellow_mean_rgb_values[:, 0],
        yellow_mean_rgb_values[:, 1],
        yellow_mean_rgb_values[:, 2],
        color="yellow",
    )
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    plt.legend()
    plt.show()
