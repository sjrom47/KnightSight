import cv2
import numpy as np
from utils import show_image, load_images
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


class ColorClassifier:
    def __init__(self):
        self.model = LinearSVC()
        self._yellow_mean_rgb_values = []
        self._blue_mean_rgb_values = []

    def add_yellow_mean_rgb_value(self, value):
        self._yellow_mean_rgb_values.append(value)

    def add_blue_mean_rgb_value(self, value):
        self._blue_mean_rgb_values.append(value)

    def train(self):
        X = self._yellow_mean_rgb_values + self._blue_mean_rgb_values
        y = [0] * len(self._yellow_mean_rgb_values) + [1] * len(
            self._blue_mean_rgb_values
        )
        self.model.fit(X, y)

    def predict(self, value):
        return self.model.predict([value])
