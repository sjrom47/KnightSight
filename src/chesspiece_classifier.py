from image_classifier import ImageClassifier
from bow import BoW
from config import *
from color_classifier import ColorClassifier
import cv2
import numpy as np


class ChessPieceClassifier:
    def __init__(
        self,
        piece_classifier_path=CLASSIFIER_DIR,
        color_classifier_path=COLOR_CLASSIFIER_DIR,
    ):
        bow = BoW()
        bow.load_vocabulary(f"{piece_classifier_path}/vocabulary")
        self._piece_classifier = ImageClassifier(bow)
        self._piece_classifier.load(f"{piece_classifier_path}/classifier")
        self._color_classifier = ColorClassifier()
        self._color_classifier.load(f"{color_classifier_path}/color_classifier")
        self._mser = cv2.MSER_create()
        self._label_piece_conversion = {
            label: piece for piece, label in self._piece_classifier.labels.items()
        }
        self._label2piece = lambda label: (
            self._label_piece_conversion[label] if label is not None else None
        )

    @property
    def piece_classifier(self):
        return self._piece_classifier

    @property
    def color_classifier(self):
        return self._color_classifier

    @property
    def piece_labels(self):
        return self._piece_classifier.labels

    @property
    def color_labels(self):
        return {"white": 0, "black": 1}

    def classify(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if np.std(gray_image) / np.mean(gray_image) < 0.12:
            return (None, None)
        return (
            self._label2piece(self.classify_piece(image)),
            self.classify_color(image)[0],
        )

    def classify_piece(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            return self._piece_classifier.predict_single(image)
        except ValueError:
            return None

    def classify_color(self, image):
        return self._color_classifier.predict(image)

    def classify_batch(self, images):
        return [self.classify(image) for image in images]
