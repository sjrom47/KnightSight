import numpy as np
import cv2
from typing import List
import glob
import os


def non_max_suppression(img, theta):

    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.float32)

    # converting radians to degree
    angle = theta * 180.0 / np.pi  # max -> 180, min -> -180
    angle[angle < 0] += 180  # max -> 180, min -> 0

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                r = img[i, j - 1]
                q = img[i, j + 1]
            # angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                r = img[i + 1, j + 1]
                q = img[i - 1, j - 1]
            # angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                r = img[i - 1, j]
                q = img[i + 1, j]
            # angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                r = img[i - 1, j + 1]
                q = img[i + 1, j - 1]

            if (img[i, j] >= q) and (img[i, j] >= r):
                Z[i, j] = img[i, j]
            else:
                Z[i, j] = 0
    return Z


def nothing(x):
    pass


def get_hsv_color_ranges(image: np.array):

    # Create a window
    cv2.namedWindow("image")

    # Create trackbars for color change
    cv2.createTrackbar("HMin", "image", 0, 255, nothing)
    cv2.createTrackbar("SMin", "image", 0, 255, nothing)
    cv2.createTrackbar("VMin", "image", 0, 255, nothing)
    cv2.createTrackbar("HMax", "image", 0, 255, nothing)
    cv2.createTrackbar("SMax", "image", 0, 255, nothing)
    cv2.createTrackbar("VMax", "image", 0, 255, nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos("HMax", "image", 255)
    cv2.setTrackbarPos("SMax", "image", 255)
    cv2.setTrackbarPos("VMax", "image", 255)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    output = image
    wait_time = 33

    while 1:

        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos("HMin", "image")
        sMin = cv2.getTrackbarPos("SMin", "image")
        vMin = cv2.getTrackbarPos("VMin", "image")

        hMax = cv2.getTrackbarPos("HMax", "image")
        sMax = cv2.getTrackbarPos("SMax", "image")
        vMax = cv2.getTrackbarPos("VMax", "image")

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if (
            (phMin != hMin)
            | (psMin != sMin)
            | (pvMin != vMin)
            | (phMax != hMax)
            | (psMax != sMax)
            | (pvMax != vMax)
        ):
            print(
                "(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)"
                % (hMin, sMin, vMin, hMax, sMax, vMax)
            )
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        cv2.imshow("image", output)
        cv2.resizeWindow("image", 500, 300)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def opencv_load_images(filenames: List) -> List:
    """
    Load images cv2.imread function (BGR)
    """
    return [cv2.imread(filename) for filename in filenames]


def show_image(img: np.array, img_name: str = "Image", resize=False) -> None:
    """
    Display an image using OpenCV.

    Args:
        img (np.array): the image to display.
        img_name (str, optional): The name of the window. Defaults to "Image".
        resize (bool, optional): If the image is too big we can resize it. Defaults to False.
    """
    if resize:
        img = cv2.resize(img, (800, 600))
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_images(path: str) -> List:
    """
    Loads images from a given path.

    Args:
        path (str): The path to the images.

    Returns:
        List: a list of images.
    """
    paths_to_images = glob.glob(path)
    imgs = opencv_load_images(paths_to_images)
    return imgs


def save_image(img: np.array, path: str, img_name: str) -> None:
    """
    Save an image to a given path.

    Args:
        img (np.array): The image to save.
        path (str): The path to save the image.
        img_name (str): The name of the image.
    """
    print(path)
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(path + img_name + ".jpg", img)
    print(f"Image saved to {path}")
