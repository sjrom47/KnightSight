import cv2
import numpy as np
from typing import List, Tuple


def warp_chessboard_image(img: np.array, grid: List, grid_size: Tuple) -> np.array:
    """
    Warp the image to a top-down view
    """
    grid = grid.reshape(-1, 2)
    ideal_grid = np.mgrid[0 : grid_size[0] + 1, 0 : grid_size[1] + 1] * 100 + 50
    ideal_grid = ideal_grid.T.reshape(-1, 2)
    img_size = (img.shape[1], img.shape[0])
    # M = cv2.getPerspectiveTransform(np.float32(grid), np.float32(dst))
    M, _ = cv2.findHomography(np.float32(grid), np.float32(ideal_grid), cv2.RANSAC)
    warped = cv2.warpPerspective(
        img,
        M,
        ((grid_size[0] + 1) * 100, (grid_size[1] + 1) * 100),
        flags=cv2.INTER_LINEAR,
    )
    return warped
