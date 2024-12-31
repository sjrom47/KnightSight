from chessboard_detection import find_chessboard_corners
from warped_img import warp_chessboard_image
from typing import List, Tuple
from utils import show_image, save_image, load_images
import cv2
from config import *
import os
import time
from utils import split_image_into_squares


def show_image_to_label(image):
    cv2.imshow("image", image)
    for i, piece in enumerate(PIECE_TYPES):
        print(f"Press {i} to label {piece}")
    key = cv2.waitKey(0)
    print(chr(key))
    try:
        piece = PIECE_TYPES[int(chr(key))]
        os.makedirs(f"{LABELED_IMAGES_DIR}/{piece}", exist_ok=True)
        print(f"You selected {piece}")
        existing_files = [
            f
            for f in os.listdir(f"{LABELED_IMAGES_DIR}/{piece}")
            if f.startswith(piece) and f.endswith(".jpg")
        ]
        existing_numbers = sorted(
            [int(f.split("_")[1].split(".")[0]) for f in existing_files]
        )
        file_number = 1
        for num in existing_numbers:
            if num != file_number:
                break
            file_number += 1
        print(f"Saving as {piece}_{file_number:03d}")
        save_image(
            image, f"{LABELED_IMAGES_DIR}/{piece}/", f"/{piece}_{file_number:03d}"
        )
    except Exception as e:
        print(e)
        print("You selected nothing in the square")
    cv2.destroyAllWindows()


def label_image_set(image_dir: str, grid_size: Tuple) -> List:
    """
    Label the image set
    """
    images = load_images(image_dir)
    print(images)
    for image in images:
        # ? See if resizing is beneficial or not
        image_shape = (image.shape[1] // 2, image.shape[0] // 2)
        image = cv2.resize(image, image_shape, interpolation=cv2.INTER_AREA)
        # show_image(image, resize=True)
        _, grid = find_chessboard_corners(image, sigma=2)
        warped_img, _ = warp_chessboard_image(image, grid, grid_size)
        show_image(warped_img, resize=True)
        squares = split_image_into_squares(warped_img, grid_size)
        for square in squares:
            show_image_to_label(square)
            time.sleep(0.1)


if __name__ == "__main__":
    label_image_set(UNLABELED_IMAGES_DIR + "/*.jpg", (8, 8))
