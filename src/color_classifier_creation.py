import cv2
import numpy as np
from utils import *
from color_classifier import ColorClassifier
from config import *


def create_color_classifier(white_pieces, black_pieces, save=False):
    color_classifier = ColorClassifier()
    for image in white_pieces:
        color_classifier.add_white_mean_rgb_value(np.mean(image, axis=(0, 1)))
    for image in black_pieces:
        color_classifier.add_black_mean_rgb_value(np.mean(image, axis=(0, 1)))
    color_classifier.train()
    if save:
        color_classifier.save("color_classifier")
    return color_classifier


def load_training_set(path=COLOR_LABELED_IMAGES_DIR, resize=True):
    training_set = {"black": [], "white": []}
    for color in training_set:
        imgs = load_images(f"{path}/{color}/*jpg")
        if resize:
            imgs = [
                cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA) for img in imgs
            ]
        training_set[color] = imgs

    return training_set


if __name__ == "__main__":
    training_set = load_training_set()
    color_classifier = create_color_classifier(
        training_set["white"], training_set["black"], save=True
    )
    print("Color classifier created and saved.")
