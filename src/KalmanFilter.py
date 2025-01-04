from utils import load_video, get_hsv_color_ranges
import numpy as np
import copy
import cv2
from config import *
from utils import show_image


class KalmanFilter:
    def __init__(
        self, min_range_hsv=HAND_HSV_RANGES[0], max_range_hsv=HAND_HSV_RANGES[1]
    ):
        self._min_range_hsv = min_range_hsv
        self._max_range_hsv = max_range_hsv

        # Inicialización del filtro de Kalman
        self._kf = cv2.KalmanFilter(4, 2)
        self._kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-5

        self._track_window = None

        self._crop_hist = None
        self._term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)

    def get_bounding_box(self, frame):

        x, y, w, h = self.get_minimum_rectangle(frame)
        draw_frame = frame.copy()
        cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return x, y, w, h

    def get_minimum_rectangle(self, frame):
        non_null_pixels = np.argwhere(frame.T > 0)
        x, y, w, h = cv2.boundingRect(non_null_pixels)

        return x, y, w, h

    def clear(self):
        self._track_window = None
        self._crop_hist = None

    def initialize(self, frame, mog_frame):
        x, y, w, h = self.get_bounding_box(mog_frame)
        if w * h < 1000 or w * h > 200000:
            return
        self._track_window = (x, y, w, h)
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        # Inicialización del estado posterior
        self._kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)

        # Histogram calculation
        crop = frame[y : y + h, x : x + w].copy()
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # show_image(hsv_crop)
        mask = cv2.inRange(hsv_crop, self._min_range_hsv, self._max_range_hsv)
        # show_image(mask)
        crop_hist = cv2.calcHist([hsv_crop], [0], mask, [180], [0, 180])
        cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)

        self._crop_hist = crop_hist

    def predict(self, frame):
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        img_bproject = cv2.calcBackProject([img_hsv], [0], self._crop_hist, [0, 180], 1)

        ret, self._track_window = cv2.meanShift(
            img_bproject, self._track_window, self._term_crit
        )
        x_, y_, w_, h_ = self._track_window
        c_x, c_y = x_ + w_ // 2, y_ + h_ // 2

        # Predicción y corrección
        prediction = self._kf.predict()
        measurement = np.array([[c_x], [c_y]], np.float32)
        self._kf.correct(measurement)
        points = (x_, y_, w_, h_)
        points = self.rect_to_points(*points)

        return points, prediction[:2].reshape(1, 2)

    def rect_to_points(self, x, y, w, h):
        return np.array([[(x, y), (x + w, y)], [(x, y + h), (x + w, y + h)]])


if __name__ == "__main__":
    videopath = "data/other_data/videos/video_test_1.mp4"
    frames = load_video(videopath)

    # This ranges are created for the hand of the player
    min_range = (0, 63, 131)
    max_range = (255, 255, 255)

    kf = KalmanFilter(min_range, max_range)
    kf.initialize(frames)
    frames, predictions = kf.predict(frames)
