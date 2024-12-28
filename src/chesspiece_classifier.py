import cv2
import numpy as np
from image_classifier import ImageClassifier
from bow import BoW
from config import *


class ChessPieceClassifier:
    def __init__(self, classifier_path=CLASSIFIER_DIR):
        bow = BoW.load(f"{classifier_path}/vocabulary")
        self._piece_classifier = ImageClassifier(bow)
        self._piece_classifier.load(f"{classifier_path}/classifier")

    @property
    def piece_classifier(self):
        return self._piece_classifier

    def classify(self, image):
        return (self.classify_piece(image), self.classify_color(image))

    def classify_piece(self, image):
        return self._piece_classifier.predict(image)

    def classify_color(self, image):
        pass

    def classify_batch(self, images):
        return [self.classify(image) for image in images]
