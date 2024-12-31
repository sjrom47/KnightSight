import cv2
import numpy as np


class Subtractor:
    def __init__(self, image=None):
        self._image = image

    def subtract(self, frame):
        if self._image is None:
            self.set_image(frame)
            return None
        else:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(self._image, frame)
            self.set_image(frame)
            return diff

    def set_image(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self._image = image

    def identify_moved_squares(self, diffs, grid_size=(8, 8)):
        square_sum = []
        for diff in diffs:
            square_sum.append(np.sum(diff))
        top5_squares = np.argsort(square_sum, axis=None)[::-1][:5]
        ratio23 = square_sum[top5_squares[1]] / square_sum[top5_squares[2]]
        ratio12 = square_sum[top5_squares[0]] / square_sum[top5_squares[1]]
        ratio45 = square_sum[top5_squares[3]] / square_sum[top5_squares[4]]
        #! Castling still not implemented
        print(square_sum[top5_squares[0]] >10000,ratio23 / ratio12 > 2, square_sum[top5_squares[0]], top5_squares[0])
        if ratio23 / ratio12 > 2 or square_sum[top5_squares[0]] >300000:
            transformed_squares = self.transform_to_board_coords(
                top5_squares[:2], grid_size
            )

            return transformed_squares
        elif ratio45 > 2:
            transformed_squares = self.transform_to_board_coords(
                top5_squares[:4], grid_size
            )
            return transformed_squares
        else:
            return None

    def transform_to_board_coords(self, squares, grid_size=(8, 8)):
        board_coords = []
        for square in squares:
            x = square // grid_size[0]
            y = square % grid_size[0]
            board_coords.append((x, y))
        return board_coords
