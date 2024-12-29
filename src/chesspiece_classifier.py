from image_classifier import ImageClassifier
from bow import BoW
from config import *
from color_classifier import ColorClassifier


class ChessPieceClassifier:
    def __init__(
        self,
        piece_classifier_path=CLASSIFIER_DIR,
        color_classifier_path=COLOR_CLASSIFIER_DIR,
    ):
        bow = BoW.load(f"{piece_classifier_path}/vocabulary")
        self._piece_classifier = ImageClassifier(bow)
        self._piece_classifier.load(f"{piece_classifier_path}/classifier")
        self._color_classifier = ColorClassifier()
        self._color_classifier.load(f"{color_classifier_path}/color_classifier")

    @property
    def piece_classifier(self):
        return self._piece_classifier

    @property
    def color_classifier(self):
        return self._color_classifier

    def classify(self, image):
        return (self.classify_piece(image), self.classify_color(image))

    def classify_piece(self, image):
        return self._piece_classifier.predict(image)

    def classify_color(self, image):
        return self._color_classifier.predict(image)

    def classify_batch(self, images):
        return [self.classify(image) for image in images]
