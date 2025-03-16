from utils import *
from chessboard_detection import find_chessboard_corners
from chesspiece_classifier import ChessPieceClassifier
from config import *
from warped_img import warp_chessboard_image, unwarp_points, get_ideal_grid
from tracking import Tracker
from GMM import GMM_filter
from visual_board import VisualBoard
from Subtractor import Subtractor
from enum import Enum, auto
from KalmanFilter import KalmanFilter
from InfoToStockfish import ChessBot
import pygame
import time
from security_system import SecuritySystem


class KnightSightState(Enum):
    CORNER_DETECTION = auto()
    HAND = auto()
    TRACKING = auto()


class KnightSight:
    def __init__(
        self,
        hand_threshold=10000,
        board_size=(8, 8),
        piece_classifier_path=CLASSIFIER_DIR,
        color_classifier_path=COLOR_CLASSIFIER_DIR,
        include_fps=False,
        fix_fps=20,
    ):
        self._classifier = ChessPieceClassifier(
            piece_classifier_path, color_classifier_path
        )
        self._tracker = Tracker()
        self._gmm_filter = GMM_filter(history=120)
        self._visual_board = VisualBoard()
        self._subtractor = Subtractor()
        self._chess_bot = ChessBot(STOCKFISH_PATH)
        self._kalman = KalmanFilter()
        self._board_size = board_size
        self._hand_threshold = hand_threshold
        self._state = None
        self._corners = None
        self._threshold = hand_threshold
        self._piece2int = {piece: i + 1 for i, piece in enumerate(PIECE_TYPES)}
        self._objects_present = False
        self._include_fps = include_fps
        self._fix_fps = fix_fps
        self._moved_piece = None
        self._remaining_draw_piece_frames = 0
        self._processed_frames = 0

    @property
    def classifier(self):
        return self._classifier

    @property
    def tracker(self):
        return self._tracker

    @property
    def filter(self):
        return self._filter

    @property
    def visual_board(self):
        return self._visual_board

    @property
    def subtractor(self):
        return self._subtractor

    @property
    def board_size(self):
        return self._board_size

    def initialise_first_frame(self, img, override=False):
        if not override:
            _, grid = find_chessboard_corners(img)

            warped_img, M = warp_chessboard_image(img, grid)
            squares_imgs = split_image_into_squares(warped_img, self._board_size)
            # for img in squares_imgs:
            #     print(self._classifier.classify(img))
            # show_image(img)

            labels = self._classifier.classify_batch(squares_imgs)

            temp_board_state = self.labels2board(labels)
            temp_board = VisualBoard()
            temp_board.set_initial_state(temp_board_state)
            print(
                {j: i + 1 for i, j in self._classifier._label_piece_conversion.items()}
            )
            print(temp_board)

            all_correct = input("Is the board correct? (y/n) ")
            while all_correct.lower() != "y":
                row, col = input(
                    "Enter the row and column of the piece to change: "
                ).split()
                piece = input("Enter the piece: ")
                color = input("Enter the color (0:white, 1:black): ")
                try:
                    color = int(color)
                except ValueError:
                    color = 0
                int_piece = self.piece_and_color_to_int(piece, color)

                temp_board.replace_piece(int(row), int(col), int_piece)
                print(
                    {
                        j: i + 1
                        for i, j in self._classifier._label_piece_conversion.items()
                    }
                )
                print(temp_board)
                all_correct = input("Is the board correct? (y/n) ")

            self._visual_board.set_initial_state(temp_board.state)
        else:
            default_board = [
                [-4, -2, -3, -5, -6, -3, -2, -4],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [4, 2, 3, 5, 6, 3, 2, 4],
            ]
            self._visual_board.set_initial_state(default_board)

    def labels2board(self, labels):
        temp_board = []
        for row in range(self._board_size[0]):
            row_pieces = []
            for col in range(self._board_size[1]):
                int_piece = self.piece_and_color_to_int(
                    *labels[row * self._board_size[1] + col]
                )

                row_pieces.append(int_piece)
            temp_board.append(row_pieces)
        return temp_board

    def piece_and_color_to_int(self, piece, color):
        piece_value = self._piece2int.get(piece, None)
        # color = 0 if color == "white" else 1
        if piece_value is not None:
            int_piece = (1 - 2 * color) * piece_value
        else:
            int_piece = 0
        return int_piece

    def corner_detection(self, img):
        _, grid = find_chessboard_corners(img, sigma=1.5)
        warped_img, M = warp_chessboard_image(img, grid)
        ideal_grid = get_ideal_grid(self._board_size)
        self._corners = unwarp_points(ideal_grid, M)
        self._tracker.set_up_first_frame(img, self._corners)
        new_warped_img, M = warp_chessboard_image(img, self._corners)
        return new_warped_img

    def process_frame(self, img):

        start_time = time.time()
        if self._corners is None:
            warped_img = self.corner_detection(img)
        else:
            warped_img, M = warp_chessboard_image(img, self._corners)
            warped_masked_img = self._gmm_filter.apply(warped_img)
            self._objects_present = sum(sum(warped_masked_img)) > self._hand_threshold
        # show_image(warped_masked_img)
        draw_img = img.copy()

        if self._state == KnightSightState.HAND:
            if not self._objects_present:
                self._state = KnightSightState.CORNER_DETECTION
            else:
                self._state = KnightSightState.HAND
        elif self._state == KnightSightState.TRACKING:
            if self._objects_present:
                self._state = KnightSightState.HAND
            else:
                self._state = KnightSightState.TRACKING
        elif self._state == KnightSightState.CORNER_DETECTION:
            self._state = KnightSightState.TRACKING
        else:
            self._state = KnightSightState.CORNER_DETECTION

        if self._state == KnightSightState.CORNER_DETECTION:
            self._kalman.clear()
            self._gmm_filter.reset()

            warped_img = self.corner_detection(img)
            difference = self._subtractor.subtract(warped_img)
            if difference is not None:
                # show_image(difference)
                square_diffs = split_image_into_squares(difference, self._board_size)
                moved_squares = self._subtractor.identify_moved_squares(square_diffs)

                if moved_squares is not None:
                    end_ind = self._visual_board.make_move(*moved_squares)
                    self._moved_piece = moved_squares[end_ind]
                    self._remaining_draw_piece_frames = 15
                    #! This has to receive confirmation in final version
                    print(
                        self._chess_bot.check_legal_move(
                            self.visual_board._temp_state,
                            "w" if self._visual_board.playing == 1 else "b",
                            "-",
                            0,
                            1,
                        )
                    )
                    self._visual_board.confirm_move()

            warped_masked_img = self._gmm_filter.apply(warped_img)

        elif self._state == KnightSightState.TRACKING:
            corners = self._tracker.track(img)
            if len(corners) == self._board_size[0] * self._board_size[1]:
                self._corners = corners

        elif self._state == KnightSightState.HAND:

            if self._kalman._track_window is None:
                # Copy the mask for processing
                warped_masked_img_copy = warped_masked_img.copy()

                # Apply morphological operations
                erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                warped_masked_img_copy = cv2.erode(
                    warped_masked_img_copy, erosion_kernel
                )
                warped_masked_img_copy = cv2.dilate(
                    warped_masked_img_copy, dilation_kernel
                )

                # Add explicit thresholding to remove shadows (values below 200 are considered shadows)
                _, warped_masked_img_copy = cv2.threshold(
                    warped_masked_img_copy,
                    100,  # Threshold value - adjust based on your lighting conditions
                    255,
                    cv2.THRESH_BINARY,
                )

                # Debug visualization
                # show_image(warped_masked_img_copy, resize=False)

                # Dynamic threshold calculation remains the same
                area = warped_masked_img_copy.shape[0] * warped_masked_img_copy.shape[1]
                dynamic_threshold = area * 0.15
                object_present = self.check_for_objects(
                    warped_masked_img_copy, threshold=dynamic_threshold
                )
                print(
                    f"Object present: {object_present}, Threshold: {dynamic_threshold}"
                )
                if object_present:
                    self._kalman.initialize(warped_img, warped_masked_img)
            else:
                points, prediction = self._kalman.predict(warped_img)
                points = points.reshape(-1, 2)
                if prediction is not None:

                    prediction = unwarp_points(prediction, M)
                    draw_img = self.draw_points_on_frame(
                        prediction, draw_img, (0, 255, 0)
                    )
                if points is not None:
                    points = unwarp_points(points, M)
                    draw_img = self.draw_rectangle_on_frame(
                        points, draw_img, (255, 0, 0)
                    )

        draw_img = self.draw_points_on_frame(self._corners, draw_img)
        end_process_time = time.time()
        if self._fix_fps:
            elapsed_time = end_process_time - start_time
            if elapsed_time < 1 / self._fix_fps:
                time.sleep(1 / self._fix_fps - elapsed_time)
        if self._include_fps:
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(
                draw_img,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        if self._remaining_draw_piece_frames > 0:
            rectangle_position = self.get_square_corners(*self._moved_piece)
            self.draw_rectangle_on_frame(rectangle_position, draw_img, (0, 255, 0))
            self._remaining_draw_piece_frames -= 1
        cv2.imshow("Tracking", draw_img)
        self._processed_frames += 1
        # cv2.waitKey(10)

    def draw_points_on_frame(self, corners, frame, color=(0, 0, 255)):
        for corner in corners:
            cv2.circle(frame, tuple(int(j) for j in corner), 5, color, -1)

        return frame

    def draw_rectangle_on_frame(self, rectangle, frame, color=(0, 0, 255)):
        p1, p2, p3, p4 = rectangle
        cv2.line(frame, tuple(int(i) for i in p1), tuple(int(i) for i in p2), color, 2)
        cv2.line(frame, tuple(int(i) for i in p2), tuple(int(i) for i in p4), color, 2)
        cv2.line(frame, tuple(int(i) for i in p4), tuple(int(i) for i in p3), color, 2)
        cv2.line(frame, tuple(int(i) for i in p3), tuple(int(i) for i in p1), color, 2)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        return frame

    def get_square_corners(self, row, col):
        """Convert chess square coordinates to four corner points"""
        if self._corners is None or len(self._corners) < 81:  # 9x9 grid for 8x8 board
            return None

        # Reshape corners to grid format
        grid_corners = np.array(self._corners).reshape(
            self._board_size[0] + 1, self._board_size[1] + 1, 2
        )

        # Get the four corners of the square
        p1 = grid_corners[row][col]
        p2 = grid_corners[row][col + 1]
        p3 = grid_corners[row + 1][col + 1]
        p4 = grid_corners[row + 1][col]

        return np.array([p1, p2, p4, p3])  # clockwise order

    # def check_for_objects(self, img, threshold=None):
    #     if threshold is None:
    #         threshold = self._threshold
    #     if len(img.shape) == 3:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         # img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    #     # print(sum(sum(img)))
    #     result = sum(sum(img))
    #     print(result)
    #     return result > threshold

    def check_for_objects(self, img, threshold=None, min_area=1000):
        """Check for objects using contour analysis and convex hull"""
        if threshold is None:
            threshold = self._threshold

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Ensure binary image
        _, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the convex hull
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)

        # Calculate bounding box as alternative
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox_area = w * h

        # Calculate area ratio (area of contour / area of hull)
        # This helps distinguish hand-like shapes from simple blobs
        contour_area = cv2.contourArea(largest_contour)
        solidity = contour_area / hull_area if hull_area > 0 else 0

        # Debug output
        print(
            f"Hull area: {hull_area}, Bounding box area: {bbox_area}, Solidity: {solidity:.2f}"
        )

        # Check both size and shape characteristics
        is_large_enough = bbox_area > threshold
        is_hand_shaped = 0.3 < solidity < 0.9  # Hand typically has this solidity range

        print(f"Is large enough: {is_large_enough}, Is hand shaped: {is_hand_shaped}")

        return is_large_enough  # and is_hand_shaped


def main(image, video):
    knight_sight.initialise_first_frame(*image, override=True)
    for frame in video:

        knight_sight.process_frame(frame)
        # print(knight_sight.visual_board)


def draw_board(board):
    for row in range(8):
        for col in range(8):
            color = (255, 255, 255) if (row + col) % 2 == 0 else (125, 135, 150)
            pygame.draw.rect(
                screen,
                color,
                (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
            )
            piece = board.state[row][col]
            if piece:
                piece_image = pieces_images.get(piece)
                screen.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))


if __name__ == "__main__":
    # security_system = SecuritySystem()
    # security_system.run()
    print("Starting KnightSight...")
    knight_sight = KnightSight(include_fps=False)
    # Initialize the camera
    print("Place the camera pointing to the chessboard")

    first_frame = load_images("data/unlabeled_data/*")[0]
    video = load_video("data/test_video2.mp4")

    knight_sight.initialise_first_frame(first_frame, override=True)

    print("KnightSight has been set up.")
    print("Press 'q' to quit")

    pygame.init()

    WIDTH, HEIGHT = 600, 600  # Size of the window
    SQUARE_SIZE = WIDTH // 8  # Size of each square

    # Load images for pieces
    pieces_images = {
        1: pygame.image.load("data/Piezas/wp.png"),
        2: pygame.image.load("data/Piezas/wn.png"),
        3: pygame.image.load("data/Piezas/wb.png"),
        4: pygame.image.load("data/Piezas/wr.png"),
        5: pygame.image.load("data/Piezas/wq.png"),
        6: pygame.image.load("data/Piezas/wk.png"),
        -1: pygame.image.load("data/Piezas/bp.png"),
        -2: pygame.image.load("data/Piezas/bn.png"),
        -3: pygame.image.load("data/Piezas/bb.png"),
        -4: pygame.image.load("data/Piezas/br.png"),
        -5: pygame.image.load("data/Piezas/bq.png"),
        -6: pygame.image.load("data/Piezas/bk.png"),
    }

    for key, image in pieces_images.items():
        pieces_images[key] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))

    # Set up the Pygame screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess Game")

    for frame in video:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow("frames", frame)

        knight_sight.process_frame(frame)
        draw_board(knight_sight.visual_board)
        pygame.display.flip()

    cv2.destroyAllWindows()
