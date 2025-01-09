PIECE_TYPES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
SECURITY_CODE = ["white", "black", "white", "white"]
HAND_HSV_RANGES = [(0, 80, 20), (180, 170, 160)]
PIECE_COLOR_MASKS = {
    "white": [(20, 85, 100), (35, 255, 255)],
    "black": [(90, 0, 90), (105, 255, 255)],
}
LABELED_IMAGES_DIR = "./data/labeled_data"
UNLABELED_IMAGES_DIR = "./data/unlabeled_data"
DATASET_DIR = "./data/labeled_data"
CLASSIFIER_DIR = "./data/classifier"
COLOR_CLASSIFIER_DIR = f"./data/color_classifier"
COLOR_LABELED_IMAGES_DIR = f"./data/color_labels"
VIDEOS_DIR = "./data/other_data/videos"
STOCKFISH_PATH = "./stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"
