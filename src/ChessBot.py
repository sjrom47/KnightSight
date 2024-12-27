from stockfish import Stockfish
import chess

class ChessBot:
    """
    A ChessBot class that uses the Stockfish chess engine to make moves based on the current board state.

    Attributes:
    ----------
    board : chess.Board
        The current board state for the game.
    engine : Stockfish
        Instance of the Stockfish chess engine used to analyze and determine the best moves.

    Methods:
    -------
    __init__(stockfish_path: str, depth: int = 20, elo_rating: int = -1)
        Initializes the ChessBot instance with a Stockfish engine, board, depth, and optional ELO rating.
    
    move() -> chess.Board or None
        Executes the best move according to Stockfish and updates the board. Returns the updated board state if a move
        is possible; otherwise, returns None.
    
    set_board(board: chess.Board)
        Sets the current board state for the bot to the specified `board`.
    """

    def __init__(self, stockfish_path: str, depth: int = 20, elo_rating: int = -1):
        """
        Initializes the ChessBot with a Stockfish engine and chess board.

        Parameters:
        ----------
        stockfish_path : str
            The file path to the Stockfish executable.
        depth : int, optional
            The depth level for move calculations, which determines how deeply the engine analyzes each move.
            Defaults to 20.
        elo_rating : int, optional
            Sets an ELO rating limit for the Stockfish engine to simulate a specific skill level. If not specified
            or set to -1, the engine runs at full strength.
        """
        self.board = chess.Board()
        self.engine = Stockfish(stockfish_path)
        self.engine.set_skill_level(depth)

        if elo_rating != -1:
            self.engine.set_elo_rating(elo_rating)

    def move(self):
        """
        Calculates and makes the best move for the current board state according to the Stockfish engine.

        Returns:
        -------
        chess.Board or None
            The updated board state after the move is made. If no move is possible (e.g., game over), returns None.
        """
        fen = self.board.fen()
        self.engine.set_fen_position(fen)

        stockfish_move = self.engine.get_best_move()

        if stockfish_move:
            self.board.push_uci(stockfish_move)
            return self.board

        return None

    def set_board(self, board: chess.Board):
        """
        Updates the ChessBot's current board state to the specified board.

        Parameters:
        ----------
        board : chess.Board
            A `chess.Board` instance representing the new board state to be set.
        """
        self.board = board
