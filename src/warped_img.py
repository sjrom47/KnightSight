import cv2
import numpy as np
from typing import List


def warp_chessboard_image(img: np.array, src: List, dst: List) -> np.array:
    """
    Warp the image to a top-down view
    """
    img_size = (img.shape[1], img.shape[0])
    # M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    M, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped
