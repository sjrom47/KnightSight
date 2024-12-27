import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from bow import BoW
from dataset import Dataset
from image_classifier import ImageClassifier
import time
from tqdm import tqdm
import sys
import pickle
import itertools
from config import *


def load_dataset(dataset_path):
    training_set = Dataset.load(dataset_path, ".jpg")
    training_set, validation_set = Dataset.split(training_set, 0.7)
    return training_set, validation_set


def create_bow(
    training_set,
    feature_extractor="SIFT",
    vocabulary_size=100,
    iterations=100,
    save=False,
):
    bow = BoW()
    bow.build_vocabulary(training_set, feature_extractor, vocabulary_size, iterations)
    if save:
        bow.save_vocabulary(
            f"{CLASSIFIER_DIR}/vocabulary_{feature_extractor}_{vocabulary_size}.pkl"
        )
    return bow


def train_classifier(bow, training_set, iterations=100, save=False):
    image_classifier = ImageClassifier(bow)
    # Especify the args for the training method
    image_classifier.train(training_set, iterations=iterations)
    if save:
        image_classifier.save(f"{CLASSIFIER_DIR}/classifier.pkl")
    return image_classifier


def train_predict(image_classifier, training_set):
    accuracy, confusion_matrix, classification = image_classifier.predict(training_set)
    print("\nAccuracy on training set:", accuracy)
    print("\nConfusion matrix on training set:\n", confusion_matrix)
    return accuracy, confusion_matrix, classification


def test_predict(image_classifier, validation_set):
    accuracy, confusion_matrix, classification = image_classifier.predict(
        validation_set
    )
    print("\nAccuracy on validation set:", accuracy)
    print("\nConfusion matrix on validation set:\n", confusion_matrix)
    return accuracy, confusion_matrix, classification


def create_bow_and_train(
    training_set,
    validation_set=None,
    feature_extractor="SIFT",
    vocabulary_size=100,
    iterations=100,
    save=False,
):
    # training_set, validation_set = load_dataset(dataset_path)
    bow = create_bow(training_set, feature_extractor, vocabulary_size, iterations, save)
    image_classifier = train_classifier(bow, training_set, iterations, save)
    train_predict(image_classifier, training_set)
    if validation_set:
        accuracy, _, _ = test_predict(image_classifier, validation_set)
    return accuracy


def cross_validation(dataset_path, cv_params):
    training_set, validation_set = load_dataset(dataset_path)
    keys, values = zip(*cv_params.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_params = None
    best_accuracy = 0
    # Print the combinations
    for combination in combinations:
        accuracy = create_bow_and_train(training_set, validation_set, **combination)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = combination
    print(f"\nBest parameters: {best_params}")
    print(f"\nBest accuracy: {best_accuracy}")
    all_data = training_set + validation_set
    create_bow_and_train(all_data, None, **best_params, save=True)


if __name__ == "__main__":
    training_set, validation_set = load_dataset(DATASET_DIR)
    create_bow_and_train(training_set, validation_set)
