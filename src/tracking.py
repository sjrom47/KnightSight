import cv2
import numpy as np
import time
from utils import show_image


class Tracker:
    def __init__(
        self,
        winsize=(15, 15),
        maxlevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        visualize=False,
    ):
        self._earlier_frame = None
        self._earlier_corners = None
        self._winSize = winsize
        self._maxLevel = maxlevel
        self._criteria = criteria
        self._visualize = visualize

    @property
    def tracking_params(self):
        return {
            "winSize": self._winSize,
            "maxLevel": self._maxLevel,
            "criteria": self._criteria,
        }

    def set_up_first_frame(self, frame, corners):
        frame_copy = frame.copy()
        gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        self._earlier_frame = gray_frame
        self._earlier_corners = corners

    def track(self, frame):
        current_frame = frame.copy()
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_corners, st, err = cv2.calcOpticalFlowPyrLK(
            self._earlier_frame,
            current_frame,
            self._earlier_corners,
            None,
            winSize=self._winSize,
            maxLevel=self._maxLevel,
            criteria=self._criteria,
        )
        good_new = current_corners[st == 1]

        # TODO: maybe check that all points are detected
        self._earlier_frame = current_frame.copy()
        if self._visualize:
            good_old = self._earlier_corners[st == 1]
            for _, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                current_frame = cv2.line(current_frame, (a, b), (c, d), (0, 255, 0), 2)
                current_frame = cv2.circle(current_frame, (a, b), 5, (0, 0, 255), -1)
            show_image(current_frame)
        # TODO: see if we want to keep the grid structure
        self._earlier_corners = good_new.reshape(-1, 1, 2)

        return self._earlier_corners
