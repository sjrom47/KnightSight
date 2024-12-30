from image_classifier import ImageClassifier
from bow import BoW
from config import *
from color_classifier import ColorClassifier
import cv2


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

    @property
    def piece_classifier(self):
        return self._piece_classifier

    @property
    def color_classifier(self):
        return self._color_classifier

    def classify(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        regions, _ = self._mser.detectRegions(image)
        print(len(regions))
        if len(regions) < 5:
            return (0, None)
        return (self.classify_piece(image), self.classify_color(image))

    def classify_piece(self, image):
        return self._piece_classifier.predict_single(image)

    def classify_color(self, image):
        return self._color_classifier.predict(image)

    def classify_batch(self, images):
        return [self.classify(image) for image in images]
