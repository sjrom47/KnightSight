import numpy as np
from sklearn.svm import LinearSVC
import pickle
from config import *


class ColorClassifier:
    def __init__(self):
        self._model = LinearSVC()
        self._white_mean_rgb_values = []
        self._black_mean_rgb_values = []

    @property
    def white_values(self):
        return self._white_mean_rgb_values

    @property
    def black_values(self):
        return self._black_mean_rgb_values

    def add_white_mean_rgb_value(self, value):
        self._white_mean_rgb_values.append(value)

    def add_black_mean_rgb_value(self, value):
        self._black_mean_rgb_values.append(value)

    def train(self):
        X = np.array(self._white_mean_rgb_values + self._black_mean_rgb_values).reshape(
            -1, 3
        )
        y = np.array(
            [0] * len(self._white_mean_rgb_values)
            + [1] * len(self._black_mean_rgb_values)
        )

        self._model.fit(X, y)

    def predict(self, img):
        value = np.mean(img, axis=(0, 1)).reshape(1, -1)
        return self._model.predict(value)

    def save(self, filename, path=COLOR_CLASSIFIER_DIR):
        with open(f"{path}/{filename}.pickle", "wb") as f:
            pickle.dump(self._model, f)

    def load(
        self,
        filename,
    ):
        with open(f"{filename}.pickle", "rb") as f:
            self._model = pickle.load(f)
