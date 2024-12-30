import cv2
import numpy as np


class Subtractor:
    def __init__(self, image=None):
        self._image = image

    def subtract(self, frame):
        if self._image is None:
            self.set_image(frame)
            return None
        else:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(self._image, frame)
            return diff

    def set_image(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self._image = image

    def identify_moved_squares(self, diffs):
        square_sum = np.zeros_like(diffs)
        for i in range(len(diffs)):
            for j in range(len(diffs[i])):
                square_sum[i][j] = np.sum(diffs[i][j])
