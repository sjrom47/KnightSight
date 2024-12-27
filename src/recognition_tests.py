import cv2
from typing import List
import glob
import numpy as np


def opencv_load_images(filenames: List) -> List:
    """
    Load images cv2.imread function (BGR)
    """
    return [cv2.imread(filename) for filename in filenames]


def show_image(img: np.array, img_name: str = "Image"):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


paths_to_images = glob.glob("./data/photos/test*.jpg")
imgs = opencv_load_images(paths_to_images)
