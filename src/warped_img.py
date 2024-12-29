import cv2
import numpy as np
from typing import List, Tuple


def warp_chessboard_image(img: np.array, grid: List, grid_size=(8, 8)) -> np.array:
    """
    Warp the image to a top-down view
    """
    grid = grid.reshape(-1, 2)
    ideal_grid = np.mgrid[0 : grid_size[0] + 1, 0 : grid_size[1] + 1] * 100 + 50
    ideal_grid = ideal_grid.T.reshape(-1, 2)
    M, _ = cv2.findHomography(np.float32(grid), np.float32(ideal_grid), cv2.RANSAC)
    warped = cv2.warpPerspective(
        img,
        M,
        ((grid_size[0] + 1) * 100, (grid_size[1] + 1) * 100),
        flags=cv2.INTER_LINEAR,
    )
    return warped


def unwarp_points(points: np.array, homography_matrix: np.array) -> np.array:
    """
    Unwarp points using the inverse of the homography matrix
    """
    # Invert the homography matrix
    M_inv = np.linalg.inv(homography_matrix)
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    # Apply the inverse homography matrix
    unwarped_points_homogeneous = points_homogeneous @ M_inv.T
    # Convert back to Cartesian coordinates
    unwarped_points = (
        unwarped_points_homogeneous[:, :2]
        / unwarped_points_homogeneous[:, 2][:, np.newaxis]
    )
    return unwarped_points
