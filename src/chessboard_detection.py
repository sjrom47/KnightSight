import numpy as np
import copy
import cv2
from typing import List, Tuple
import skimage as ski
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import time
from scipy.spatial import cKDTree
from utils import load_images, show_image


def blur_images(imgs: List, sigma: float) -> List:
    """
    Apply Gaussian blur to a list of images.

    Args:
        imgs (List): The list of images to blur.
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        List: The list of blurred images.
    """
    blurred_imgs = copy.deepcopy(imgs)
    blurred_imgs = [cv2.GaussianBlur(image, (0, 0), sigma) for image in blurred_imgs]
    return blurred_imgs


def plot_delaunay(points: np.ndarray) -> None:
    """
    Obtain the Delaunay triangulation of a set of points and plot it.

    Args:
        points (np.ndarray): A 2D array of points with shape (n_points, 2).
    """
    # Compute the Delaunay triangulation
    tri = Delaunay(points)

    plt.triplot(points[:, 0], points[:, 1], tri.simplices)

    plt.plot(points[:, 0], points[:, 1], "o")

    plt.show()


# Define Shi-Tomasi corner detection function
def shi_tomasi_corner_detection(
    image: np.array,
    maxCorners: int,
    qualityLevel: float,
    minDistance: int,
    corner_color: tuple,
    radius: int,
    return_corners=False,
) -> np.array:
    """
    image - Input image
    maxCorners - Maximum number of corners to return.
                 If there are more corners than are found, the strongest of them is returned.
                 maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned
    qualityLevel - Parameter characterizing the minimal accepted quality of image corners.
                   The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue or the Harris function response.
                   The corners with the quality measure less than the product are rejected.
                   For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected
    minDistance - Minimum possible Euclidean distance between the returned corners
    corner_color - Desired color to highlight corners in the original image
    radius - Desired radius (pixels) of the circle
    """
    image = image.copy()
    # Input image to Tomasi corner detector should be grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Apply cv2.goodFeaturesToTrack function
    corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)

    # corners = np.int0(corners)
    corners = corners.squeeze()

    for i in corners:
        x, y = i.ravel()
        cv2.circle(image, (int(x), int(y)), radius, corner_color, -1)
    if return_corners:
        return image, corners
    return image


def symmetric_point(P: np.array, A: np.array, B: np.array) -> np.array:
    """
    Calculate the symmetric point of P with respect to the segment AB.

    Args:
        P (np.array): The point to reflect.
        A (np.array): First endpoint of the segment.
        B (np.array): Second endpoint of the segment.

    Returns:
        np.array: The symmetric point of P with respect to the segment AB.
    """
    # Calculate the midpoint M of the segment AB
    M = (A + B) / 2

    # Calculate the vector from M to P
    V = P - M

    # Reflect P across M
    P_prime = M - V

    return P_prime


def distance_point_to_segment(P, A, B):
    """
    Calculate the distance from point P to the segment AB.

    Parameters:
    P (np.array): The point from which the distance is calculated.
    A (np.array): One endpoint of the segment.
    B (np.array): The other endpoint of the segment.

    Returns:
    float: The distance from point P to the segment AB.
    """
    # Vector from A to B
    AB = B - A
    # Vector from A to P
    AP = P - A
    # Project vector AP onto AB to find the projection point
    AB_squared = np.dot(AB, AB)
    if AB_squared == 0:
        # A and B are the same point
        return np.linalg.norm(AP)
    t = np.dot(AP, AB) / AB_squared
    # Clamp t to the range [0, 1] to find the closest point on the segment
    t = max(0, min(1, t))
    # Projection point on the segment
    projection = A + t * AB
    # Distance from P to the projection point
    distance = np.linalg.norm(P - projection)
    return distance


def calculate_vector_norms(a: np.array, b: np.array, c: np.array) -> Tuple[List, List]:
    """
    This function calculates which points have to be checked during the RANSAC algorithm to
    find the grid structure of the chessboard. It calculates nine possible points that can be

    Args:
        a (np.array): first point of the triangle
        b (np.array): second point of the triangle
        c (np.array): third point of the triangle

    Returns:
        Tuple[List, List]: a list of candidate points and a list of vector norms (for the distance threshold in the kdtree)
    """

    v1 = b - a
    v2 = c - a
    v3 = c - b
    candidate_points = [
        a - v1,  # p1
        a - v2,  # p2
        b - v3,  # p3
        b + v1,  # p4
        c + v2,  # p5
        c + v3,  # p6
        symmetric_point(a, b, c),  # p7
        symmetric_point(b, a, c),  # p8
        symmetric_point(c, a, b),  # p9
    ]
    vector_norms = [
        np.linalg.norm(v1),
        np.linalg.norm(v2),
        np.linalg.norm(v3),
        np.linalg.norm(v1),
        np.linalg.norm(v2),
        np.linalg.norm(v3),
        distance_point_to_segment(a, b, c),
        distance_point_to_segment(b, a, c),
        distance_point_to_segment(c, a, b),
    ]
    return candidate_points, vector_norms


def get_chessboard_corners(corners: np.array, img=None) -> Tuple[np.array, np.array]:
    tree = cKDTree(corners)
    initial_triangle, _ = RANSAC_corners(corners, tree)

    chessboard_corners, grid = detect_chessboard_corners(
        initial_triangle, corners, img, tree
    )
    up_points = chessboard_corners.get("up", [])
    right_points = chessboard_corners.get("right", [])
    down_points = chessboard_corners.get("down", [])
    left_points = chessboard_corners.get("left", [])

    # Concatenate points in counterclockwise order
    ordered_points = (
        up_points[:-1] + left_points[:-1] + down_points[:-1] + right_points[:-1]
    )

    return np.array(ordered_points), grid


def detect_chessboard_corners(
    initial_triangle: np.array,
    corners: np.array,
    img: np.array,
    tree: cKDTree,
    board_size=(8, 8),
) -> Tuple[dict, np.array]:
    """
    This function detects the corners of the chessboard. It includes two phases. The first phase is to construct the first square
    of the chessboard. It is built using the triangle found by the RANSAC algorithm. The second phase is to expand the grid of the
    chessboard by adding rows and columns to the initial square (explained in the expand_square_grid function). It returns the edge of
    the square with the orientaions, and all the points in the chessboard (grid).

    Args:
        initial_triangle (np.array): the triangle found by the RANSAC algorithm
        corners (np.array): The corners found by the Shi-Tomasi algorithm, the contour method or any other corner detection algorithm
        img (np.array): the image where the corners are detected (in case we want to visualize the process)
        tree (cKDTree): a KDTree object to query the points
        board_size (tuple, optional): The size of the chessboard. Defaults to (8, 8).

    Returns:
        Tuple[dict, np.array]: the corners of the chessboard and the grid of points
    """
    threshold = 0.085
    longest_side_points = longest_side(initial_triangle)
    vertex = np.array(
        list(set(map(tuple, initial_triangle)) - set(map(tuple, longest_side_points)))
    )[0]

    v1 = longest_side_points[0] - vertex
    v2 = longest_side_points[1] - vertex
    last_vertex = corners[
        tree.query_ball_point(vertex + v1 + v2, threshold * np.linalg.norm(v1))
    ][0]
    grid = np.array(
        [[vertex, longest_side_points[0]], [longest_side_points[1], last_vertex]]
    )
    initial_square = {
        "up": [longest_side_points[0], vertex],
        "down": [longest_side_points[1], last_vertex],
        "left": [vertex, longest_side_points[1]],
        "right": [last_vertex, longest_side_points[0]],
    }
    chessboard_corners, grid = expand_square_grid(
        initial_square, grid, corners, tree, board_size, img
    )
    return chessboard_corners, grid


def expand_square_grid(initial_square, grid, corners, tree, board_size, img=None):
    if (
        len(initial_square["up"]) == board_size[0] + 1
        and len(initial_square["left"]) == board_size[1] + 1
    ):
        return initial_square, grid
    threshold = 0.1
    sides_order = ["up", "left", "down", "right"]
    new_square = {"up": [], "down": [], "left": [], "right": []}

    # check for not adding rows beyond the board size
    if len(initial_square["up"]) >= board_size[0] + 1:
        del new_square["right"]
        del new_square["left"]
    if len(initial_square["left"]) >= board_size[1] + 1:
        del new_square["up"]
        del new_square["down"]

    for side in new_square:
        new_square[side] = [
            corners[
                tree.query_ball_point(
                    vertex + get_vector(side, i, grid),
                    threshold * np.linalg.norm(get_vector(side, i, grid)),
                )[0]
            ]
            for i, vertex in enumerate(initial_square[side])
            if tree.query_ball_point(
                vertex + get_vector(side, i, grid),
                threshold * np.linalg.norm(get_vector(side, i, grid)),
            )
        ]

    chosen_side = max(
        new_square,
        key=lambda x: (
            len(new_square[x]) / len(initial_square[x]),
            len(initial_square[x]),
        ),
    )
    chosen_side_value = len(new_square[chosen_side]) / len(initial_square[chosen_side])

    if chosen_side_value < 1:

        chosen_side_vector = regression_vector(np.array(new_square[chosen_side]))
        other_vector = regression_vector(np.array(initial_square[chosen_side]))

        if angle_between_lines(chosen_side_vector, other_vector) > 0.025:

            median_point = np.median(new_square[chosen_side], axis=0)
            chosen_side_vector = np.array(
                [
                    np.dot([-other_vector[1], 1], [median_point[0], median_point[1]]),
                    other_vector[1],
                ]
            )

        new_square[chosen_side] = [
            (
                corners[
                    tree.query_ball_point(
                        vertex + get_vector(chosen_side, i, grid),
                        threshold * np.linalg.norm(get_vector(chosen_side, i, grid)),
                    )[0]
                ]
                if tree.query_ball_point(
                    vertex + get_vector(chosen_side, i, grid),
                    threshold * np.linalg.norm(get_vector(chosen_side, i, grid)),
                )
                else get_approx_points(chosen_side, i, grid, chosen_side_vector)
            )
            for i, vertex in enumerate(initial_square[chosen_side])
        ]

    chosen_side_index = sides_order.index(chosen_side)
    initial_square[chosen_side] = new_square[chosen_side].copy()
    initial_square[sides_order[(chosen_side_index - 1) % 4]].append(
        new_square[chosen_side][0]
    )
    initial_square[sides_order[(chosen_side_index + 1) % 4]].insert(
        0, new_square[chosen_side][-1]
    )

    if chosen_side == "up":
        grid = np.vstack(
            (np.array(new_square[chosen_side][::-1]).reshape(1, -1, 2), grid)
        )
    elif chosen_side == "down":
        grid = np.vstack((grid, np.array(new_square[chosen_side]).reshape(1, -1, 2)))
    elif chosen_side == "left":
        grid = np.hstack((np.array(new_square[chosen_side]).reshape(-1, 1, 2), grid))
    else:
        grid = np.hstack(
            (grid, np.array(new_square[chosen_side][::-1]).reshape(-1, 1, 2))
        )

    if img is not None:
        draw_img = img.copy()
        show_grid_expansion(initial_square, grid, draw_img)
    return expand_square_grid(initial_square, grid, corners, tree, board_size, img)


def angle_between_lines(v1: np.array, v2: np.array) -> float:
    return np.arccos(
        np.dot(np.array([1, v1[1]]), np.array([1, v2[1]]))
        / (np.linalg.norm([1, v1[1]]) * np.linalg.norm([1, v2[1]]))
    )


def get_approx_points(side, index, grid, new_side_vector):
    """
    Where there are no points found by the tree we approximate them by finding the intersection of the regression
    line of the new points and the regression line of the new points on the chosen side.

    Args:
        side (str): indicates the side of the square (up, down, left, right)
        index (int): indicates which row/column of the square we are approximating
        grid (np.array): The grid of points found so far
        new_side_vector (np.array): the regression vector of the new points on the chosen side (bias included)

    Returns:
        np.array: The approximated point
    """
    points = get_points_side_index(side, index, grid)
    vector = regression_vector(points)
    approx_point = line_intersection(vector, new_side_vector)

    return approx_point


def get_points_side_index(side: str, index: int, grid: np.array) -> np.array:
    """
    Gets either a row or column of the grid of points. This is used to later calculate the regression vector of the points.
    Keep in mind that grid has the usual structure instead of the positive orientation of initial_square. We want to return the
    points so that the last points are the closest to the new side. This has no current utility, but can be used to calculate the
    median distance of the closest points to the new side.

    Args:
        side (str): the side of the square (up, down, left, right)
        index (int): Which row/column of the square we want to get
        grid (np.array): The grid of points

    Returns:
        np.array: the row/column of the grid
    """
    # Remember that because of the orientation of the initial square we have to reverse some indices to get the correct points
    if side == "up":
        points = grid[:, grid.shape[1] - index - 1]
    elif side == "down":
        points = grid[:, index]
    elif side == "left":
        points = grid[index, :]
    else:
        points = grid[grid.shape[0] - index - 1, :]
    if side in ["up", "left"]:
        points = points[::-1]
    return points


# def get_median_distance(points, n=3):
#     return np.median(
#         [np.linalg.norm(points[-i - 1] - points[-i - 2]) for i in range(n)]
#     )


def regression_vector(points: np.ndarray) -> np.ndarray:
    """
    Performs linear regression to find the best fit line for a set of points.

    Args:
        points (np.ndarray): The points to fit the line to.

    Returns:
        np.ndarray: The parameters of the regression line.
    """
    X = np.hstack(
        (
            np.ones(points.shape[0]).reshape(-1, 1),
            (np.array([point[0] for point in points]).reshape(-1, 1)),
        )
    )
    y = np.array([point[1] for point in points])
    # We use the closed form solution to find the parameters of the regression line
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w


def line_intersection(v1, v2):
    """
    Finds the intersection point of two lines defined by their parameters.

    Args:
        v1 (np.array): the parameters of the first line.
        v2 (np.array): the parameters of the second line.

    Returns:
        np.array: The point of intersection.
    """
    try:
        x = (v2[0] - v1[0]) / (v1[1] - v2[1])
    except ZeroDivisionError:
        x = (v2[0] - v1[0]) / 1e-6  # Avoid division by zero just in case
    y = np.dot(v1, np.array([1, x]))
    return np.array([x, y])


def get_vector(side: str, index: int, grid: np.array) -> np.array:
    """
    Gets the vector defined by the two closest points to the side of the square we are considering.
    Keep in mind that grid has the usual structure instead of the positive orientation of initial_square,
    so we have to reverse some indices to get the correct points.

    Args:
        side (str): the side of the square (up, down, left, right)
        index (int): the row/column of the square we are considering
        grid (np.array): the grid of points

    Returns:
        np.array: the vector defined by the two closest points to the side of the square
    """
    if side == "up":
        return grid[0][grid.shape[1] - index - 1] - grid[1][grid.shape[1] - index - 1]
    elif side == "down":
        return grid[-1][index] - grid[-2][index]
    elif side == "left":
        return grid[index][0] - grid[index][1]
    else:
        return grid[grid.shape[0] - index - 1][-1] - grid[grid.shape[0] - index - 1][-2]


def longest_side(triangle: np.array) -> Tuple[np.array, np.array]:
    """
    Find the two vertices of a triangle that make the longest side.

    Parameters:
    triangle (np.array): A 3x2 array representing the vertices of the triangle.

    Returns:
    tuple: The two vertices that make the longest side.
    """
    # Calculate the distances between each pair of vertices
    distances = [
        (np.linalg.norm(triangle[0] - triangle[1]), (triangle[0], triangle[1])),
        (np.linalg.norm(triangle[1] - triangle[2]), (triangle[1], triangle[2])),
        (np.linalg.norm(triangle[2] - triangle[0]), (triangle[2], triangle[0])),
    ]

    # Find the pair with the maximum distance
    _, longest_pair = max(distances, key=lambda x: x[0])

    return longest_pair


def show_grid_expansion(initial_square, grid, img):
    """
    This function is used to visualize the process of expanding the grid of the chessboard. It shows the initial square and the grid

    Args:
        initial_square (np.array): the initial square of the chessboard
        grid (np.array): the grid of points
        img (np.array): the image where the corners are detected
    """
    draw_img = img.copy()
    for side in initial_square:
        for i in range(len(initial_square[side]) - 1):

            cv2.line(
                draw_img,
                tuple(int(j) for j in initial_square[side][i]),
                tuple(int(j) for j in initial_square[side][i + 1]),
                (0, 0, 255),
                5,
            )
        cv2.line(
            draw_img,
            tuple(int(j) for j in initial_square[side][0]),
            tuple(int(j) for j in initial_square[side][-1]),
            (0, 0, 255),
            5,
        )
    for row in grid:
        for point in row:
            cv2.circle(draw_img, tuple(int(j) for j in point), 10, (0, 255, 0), -1)
    # for corner in corners:
    #     cv2.circle(draw_img, tuple(int(j) for j in corner), 5, (255, 0, 0), -1)
    show_image(draw_img, resize=True)


def RANSAC_corners(corners: np.array, tree: cKDTree) -> Tuple[np.array, np.array]:
    """
    We use this function to find the grid pattern in our image. We start from the Delaunay triangulation of the corners
    and see which triangles have the most 'inliers' (i.e. corners that are in the grid pattern defined by the triangle).
    We consider nine points defined by the triangle and use the distance to define a dynamic threshold for the kdtree query.

    Args:
        corners (np.array): the corners found by the Shi-Tomasi algorithm, the contour method or any other corner detection algorithm
        tree (cKDTree): a KDTree object to query the points

    Returns:
        Tuple[np.array, np.array]: the triangle that defines the grid pattern and the corners that are in the grid pattern
    """

    tri = Delaunay(corners)
    triangles = corners[tri.simplices]
    threshold = 0.085
    best_triangle = None
    best_matching_points = None
    max_hits = 0

    for _ in range(10000):
        hits = 0
        random_triangle_index = np.random.choice(len(triangles))
        random_triangle = triangles[random_triangle_index]
        a, b, c = random_triangle

        candidate_points, vector_norms = calculate_vector_norms(a, b, c)

        matching_points = set()  # Use a set to avoid duplicates

        for candidate, norm in zip(candidate_points, vector_norms):
            indices = tree.query_ball_point(candidate, threshold * norm)
            if indices:
                hits += 1
                matching_points.update(indices)  # Add indices to the set

        if hits > max_hits:
            max_hits = hits
            best_triangle = random_triangle
            best_matching_points = list(matching_points)  # Convert set to list

        if hits == 9:  # Perfect match found
            return best_triangle, corners[best_matching_points]

    # If we didn't find a perfect match, return the best match found
    if best_triangle is not None:
        return best_triangle, corners[best_matching_points]
    return None, None


def sobel_processing(img: np.array, threshold=20) -> np.array:
    """
    Preprocessing of the image to make the corner detection more robust. We use the Sobel operator to find the edges of the image
    and then we apply some morphological operations to close the gaps in the edges. In our data Sobel worked better than Canny.
    The threshold of the binary image is set to 20, but it can be adjusted based on the image.

    Args:
        img (np.array): the image to preprocess
        threshold (int, optional): the threshold of the binary image. Defaults to 20.

    Returns:
        np.array: the preprocessed image
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_img = cv2.normalize(
        ski.filters.sobel(gray_img), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )

    sobel_img = cv2.morphologyEx(sobel_img, cv2.MORPH_CLOSE, None)

    _, binary_sobel_img = cv2.threshold(sobel_img, threshold, 255, cv2.THRESH_BINARY)

    sobel_img = cv2.GaussianBlur(binary_sobel_img, (0, 0), 3)
    sobel_img = abs(255 - sobel_img)
    dilations = 5
    for _ in range(dilations):
        sobel_img = cv2.dilate(sobel_img, None)

    return sobel_img


def corner_with_contours(sobel_img):
    """
    An alternative method to find the corners of the chessboard. We use the contours of the image to find the corners.

    Args:
        sobel_img (np.array): the preprocessed image

    Returns:
        np.array: the corners detected by the contour method
    """
    contours, _ = cv2.findContours(
        abs(255 - sobel_img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    corner_clusters = []
    for contour in contours:
        # Approximate the contour with a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the approximated polygon has 4 points, assume it's a square
        if len(approx) == 4:
            for point in approx:
                corner_clusters.append(point[0])

    # Convert to numpy array
    corner_clusters = np.array(corner_clusters, dtype=np.float32)

    # Group nearby points using clustering
    final_corners = []
    used_points = set()

    for i in range(len(corner_clusters)):
        if i in used_points:
            continue

        # Get current point
        current = corner_clusters[i]
        cluster_points = [current]

        # Find all points close to current point
        for j in range(len(corner_clusters)):
            if j != i and j not in used_points:
                dist = np.linalg.norm(current - corner_clusters[j])
                if dist < 75:  # Adjust this threshold based on your image
                    cluster_points.append(corner_clusters[j])
                    used_points.add(j)

        # Calculate center of cluster
        center = np.mean(cluster_points, axis=0)
        final_corners.append(center)
    final_corners = np.array(final_corners, dtype=np.float32).reshape(-1, 1, 2)
    return final_corners


def find_chessboard_corners(
    img: np.array, sigma=4, threshold=20, visualize=False
) -> Tuple[np.array, np.array]:
    """
    Apply the whole process of finding the corners of the chessboard. It includes the preprocessing of the image, the corner detection
    and the grid expansion.

    Args:
        img (np.array): the image to process
        sigma (int, optional): The standard deviation of the gaussian kernel. Defaults to 4.
        threshold (int, optional): the threshold for the binarization. Defaults to 20.
        visualize (bool, optional): whether to visualize the process. Defaults to False.

    Returns:
        Tuple: the corners of the chessboard and the grid of points
    """
    blurred_imgs = blur_images([img], sigma)
    img = blurred_imgs[0]
    # img_shape = (img.shape[1] // 2, img.shape[0] // 2)
    # img = cv2.resize(img, img_shape, interpolation=cv2.INTER_AREA)
    sobel_img = sobel_processing(img, threshold)
    sobel_img_bgr = cv2.cvtColor(sobel_img, cv2.COLOR_GRAY2BGR)

    _, corners_shi_tomasi = shi_tomasi_corner_detection(
        sobel_img_bgr, 1000, 0.1, 20, (0, 255, 0), 10, return_corners=True
    )
    # show_image(img_shi_tomasi, resize=True)
    while True:
        try:
            chessboard_corners, grid = get_chessboard_corners(
                corners_shi_tomasi, img=img if visualize else None
            )
            break
        except np.linalg.LinAlgError:
            # Sometimes the algorithm doesn't find the grid pattern, so we try again
            pass
    return chessboard_corners, grid


if __name__ == "__main__":
    imgs = load_images("./data/photos/test*.jpg")
    sigma = 4
    t0 = time.time()
    blurred_imgs = blur_images(imgs, sigma)
    print(f"Blurring took {time.time() - t0:.3f} s")
    img = blurred_imgs[-1]
    img_shape = (img.shape[1] // 2, img.shape[0] // 2)
    img = cv2.resize(img, img_shape, interpolation=cv2.INTER_AREA)
    t0 = time.time()
    sobel_img = sobel_processing(img)
    print(f"Sobel processing took {time.time() - t0:.3f} s")
    t0 = time.time()
    final_corners = corner_with_contours(sobel_img)
    print(f"Clustering took {time.time() - t0:.3f} s")

    sobel_img_bgr = cv2.cvtColor(sobel_img, cv2.COLOR_GRAY2BGR)

    t0 = time.time()
    img_shi_tomasi, corners_shi_tomasi = shi_tomasi_corner_detection(
        sobel_img_bgr, 1000, 0.1, 20, (0, 255, 0), 10, return_corners=True
    )
    t1 = time.time()
    print(f"Shi-Tomasi corner detection took {t1-t0:.3f} s")
    t0 = time.time()
    corners, close_corners = RANSAC_corners(
        corners_shi_tomasi, cKDTree(corners_shi_tomasi)
    )
    t1 = time.time()
    print(f"RANSAC took {t1-t0:.3f} s")

    t0 = time.time()
    # corners, close_corners = RANSAC_corners(
    #     final_corners.squeeze(), cKDTree(final_corners.squeeze())
    # )
    t1 = time.time()
    # print(f"RANSAC took {t1-t0:.3f} s")
    for corner in close_corners:
        cv2.circle(img_shi_tomasi, [int(i) for i in corner], 15, (255, 0, 0), -1)
    for corner in corners:
        cv2.circle(img_shi_tomasi, [int(i) for i in corner], 15, (0, 0, 255), -1)
    show_image(img_shi_tomasi, resize=True)
    t0 = time.time()
    chessboard_corners, grid = get_chessboard_corners(
        corners_shi_tomasi, sobel_img_bgr.copy()
    )
    t1 = time.time()
    print(f"Chessboard corners took {t1-t0:.3f} s")
    final_img = copy.deepcopy(img)
    for corner in chessboard_corners:
        cv2.circle(final_img, [int(i) for i in corner], 15, (0, 0, 255), -1)
    show_image(final_img, resize=True)
    t0 = time.time()
    chessboard_corners, grid = find_chessboard_corners(imgs[0])
    t1 = time.time()
    print(f"Whole process took {t1-t0:.3f} s")
    final_img = copy.deepcopy(img)
    for corner in chessboard_corners:
        cv2.circle(final_img, [int(i) for i in corner], 15, (0, 0, 255), -1)
    show_image(final_img, resize=True)
    # import cProfile

    # cProfile.run("find_chessboard_corners(imgs[-1])")
