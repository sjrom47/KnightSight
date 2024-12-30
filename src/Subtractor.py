import cv2

class Subtractor:
    def __init__(self, image = None):
        self._image = image

    def subtract(self, frame):
        diff = cv2.absdiff(self._image, frame)
        return diff
    
    def set_image(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self._image = image