import cv2
import numpy as np
from typing import List, Tuple


def warp_chessboard_image(
    img: np.array, grid: List, grid_size=(8, 8), margin=50, square_size=100
) -> np.array:
    """
    Warp the image to a top-down view
    """
    grid = grid.reshape(-1, 2)
    ideal_grid = get_ideal_grid(grid_size)
    M, _ = cv2.findHomography(np.float32(grid), np.float32(ideal_grid), cv2.RANSAC)
    warped = cv2.warpPerspective(
        img,
        M,
        (
            grid_size[0] * square_size + 2 * margin,
            grid_size[1] * square_size + 2 * margin,
        ),
        flags=cv2.INTER_CUBIC,
    )
    return warped, M


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


def get_ideal_grid(grid_size=(8, 8), margin=50, square_size=100) -> np.array:
    """
    Get the ideal grid points for a chessboard of the specified size
    """
    ideal_grid = (
        np.mgrid[0 : grid_size[0] + 1, 0 : grid_size[1] + 1] * square_size + margin
    )
    ideal_grid = ideal_grid.T.reshape(-1, 2)
    return ideal_grid
