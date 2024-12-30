import os
import cv2


class GMM_filter:
    def __init__(self, history=200, varThreshold=25, detectShadows=False):
        self._history = history
        self._varThreshold = varThreshold
        self._detectShadows = detectShadows
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._varThreshold,
            detectShadows=self._detectShadows,
        )

    @property
    def mog_params(self):
        return {
            "history": self._history,
            "varThreshold": self._varThreshold,
            "detectShadows": self._detectShadows,
        }

    def apply(self, frame):
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = self._mog2.apply(frame)
        return mask
