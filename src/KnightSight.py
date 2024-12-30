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


class KnightSightState(Enum):
    CORNER_DETECTION = auto()
    HAND = auto()
    TRACKING = auto()


class KnightSight:
    def __init__(
        self,
        hand_threshold=3000,
        board_size=(8, 8),
        piece_classifier_path=CLASSIFIER_DIR,
        color_classifier_path=COLOR_CLASSIFIER_DIR,
    ):
        self._classifier = ChessPieceClassifier(
            piece_classifier_path, color_classifier_path
        )
        self._tracker = Tracker()
        self._gmm_filter = GMM_filter()
        self._visual_board = VisualBoard()
        self._subtractor = Subtractor()
        self._board_size = board_size
        self._hand_threshold = hand_threshold
        self._state = None
        self._corners = None

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

    def initialise_first_frame(self, img):
        _, grid = find_chessboard_corners(img)

        warped_img, M = warp_chessboard_image(img, grid)
        squares_imgs = split_image_into_squares(warped_img, self._board_size)

        labels = self._classifier.classify_batch(squares_imgs)
        # TODO: labels to pieces transformation
        pieces = None
        board_state = None

        self._visual_board.set_initial_state(board_state)

    def process_frame(self, img):
        masked_img = self._gmm_filter.apply(img)
        objects_present = self.check_for_objects(masked_img)

        if self._state == KnightSightState.HAND:
            if not objects_present:
                self._state = KnightSightState.CORNER_DETECTION
            else:
                self._state = KnightSightState.HAND
        elif KnightSightState.TRACKING:
            if objects_present:
                self._state = KnightSightState.HAND
            else:
                self._state = KnightSightState.TRACKING
        elif self._state == KnightSightState.CORNER_DETECTION:
            self._state = KnightSightState.TRACKING
        else:
            self._state = KnightSightState.CORNER_DETECTION

        if self._state == KnightSightState.CORNER_DETECTION:
            _, grid = find_chessboard_corners(img)
            warped_img, M = warp_chessboard_image(img, grid)
            ideal_grid = get_ideal_grid(self._board_size)
            original_points = unwarp_points(ideal_grid, M)
            self._tracker.set_up_first_frame(warped_img, original_points)
            difference = self._subtractor.subtract(warped_img)
            # TODO: extract movements from difference
            start, end = None
            self._visual_board.make_move(start, end)

        elif self._state == KnightSightState.TRACKING:
            self._corners = self._tracker.track(img)

    def check_for_objects(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        return sum(sum(img)) > self._threshold
