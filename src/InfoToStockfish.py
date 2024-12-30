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

    def get_fen_parameters(self):
        """
        Obtiene los parámetros necesarios para generar el formato FEN desde el tablero actual.

        Returns:
        -------
        dict
            Un diccionario con los parámetros necesarios para el FEN:
            - turn: El turno actual ('w' o 'b').
            - castling: Los enroques disponibles (por ejemplo, 'KQkq' o '-').
            - en_passant: La casilla de peón al paso (por ejemplo, 'e3' o '-').
            - halfmove: Contador de medio movimientos desde la última captura o movimiento de peón.
            - fullmove: Número de movimientos completos en la partida.
        """
        turn = "w" if self.board.turn else "b"
        castling = self.board.castling_xfen()
        en_passant = (
            chess.square_name(self.board.ep_square) if self.board.ep_square else "-"
        )
        halfmove = self.board.halfmove_clock
        fullmove = self.board.fullmove_number

        return {
            "turn": turn,
            "castling": castling,
            "en_passant": en_passant,
            "halfmove": halfmove,
            "fullmove": fullmove,
        }


def board_to_fen(board, turn, castling, en_passant, halfmove="0", fullmove="1"):
    """
    Convierte un tablero de ajedrez representado como una lista de listas al formato FEN.

    :param board: Lista de listas que representa el tablero de ajedrez.
    :param turn: Turno actual ('w' para blancas, 'b' para negras).
    :param castling: String que representa los enroques disponibles ('KQkq', '-', etc.).
    :param en_passant: Casilla de peón al paso ('e3', '-', etc.).
    :param halfmove: Contador de medio movimientos (default 0).
    :param fullmove: Número de movimientos completos (default 1).
    :return: String en formato FEN.
    """

    dict_pices = {
        0: "",
        1: "P",
        2: "N",
        3: "B",
        4: "R",
        5: "Q",
        6: "K",
        -1: "p",
        -2: "n",
        -3: "b",
        -4: "r",
        -5: "q",
        -6: "k",
    }

    fen_rows = []
    for row in board:
        fen_row = ""
        empty = 0
        for cell in row:
            if cell == 0:  # Casilla vacía
                empty += 1
            else:
                if empty > 0:  # Añadir casillas vacías previas
                    fen_row += str(empty)
                    empty = 0
                fen_row += dict_pices[cell]  # Añadir la pieza
        if empty > 0:  # Añadir las casillas vacías al final de la fila
            fen_row += str(empty)
        fen_rows.append(fen_row)

    # Unir las filas con '/'
    fen_board = "/".join(fen_rows)

    # Construir el FEN completo
    fen = f"{fen_board} {turn} {castling} {en_passant} {halfmove} {fullmove}"
    return fen



def fen_to_board(fen):
    """
    Convierte un tablero en formato FEN a una lista de listas que representa el tablero.

    Parameters:
    ----------
    fen : str
        El string FEN que describe el tablero de ajedrez.

    Returns:
    -------
    list
        Una lista de listas que representa el tablero de ajedrez.
    """
    # La parte del tablero en FEN es la primera antes del espacio
    fen_board = fen.split()[0]
    board = []

    for row in fen_board.split("/"):
        board_row = []
        for char in row:
            if char.isdigit():
                # Agregar '.' para casillas vacías
                board_row.extend(["."] * int(char))
            else:
                # Agregar la pieza correspondiente
                board_row.append(char)
        board.append(board_row)

    return board


# Esto se puede cambiar en función de como sea más conveniente
# Es la supuesta detección del tablero
board = [
    [-4, -2, -3, -5, -6, -3, -2, -4],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [4, 2, 3, 5, 6, 3, 2, 4],
]

stockfish_path = "stockfish-windows-x86-64-sse41-popcnt\stockfish\stockfish-windows-x86-64-sse41-popcnt.exe"
bot1 = ChessBot(stockfish_path, depth=10, elo_rating=-1)
bot2 = ChessBot(stockfish_path, depth=10, elo_rating=-1)

# Variables globlaes del tablero iniciales
turn = "w"
castling = "KQkq"
en_passant = "-"
halfmove = "0"
fullmove = "1"

# Convierte el tablero detectado a formato que entiende el bot (fen)
fen = board_to_fen(board, turn, castling, en_passant, halfmove, fullmove)
# fen = "r1bqkbnr/pp2pppp/2n5/2Pp4/4P3/2P5/PP3PPP/RNBQKBNR w KQkq - 0 1"


running = True
while running:
    # # Simula la detección
    # board = fen_to_board(fen)

    # Una vez detectado el tablero lo pasa a formato fen
    fen = board_to_fen(board, turn, castling, en_passant, halfmove, fullmove)
    board = chess.Board(fen)
    print(board)

    legal_moves = list(board.legal_moves)

    illegal_move = True

    bot1.set_board(board)

    while illegal_move:

        temp_board = bot1.move()

        if temp_board is None:
            break

        temp_move_legal = False
        for move in legal_moves:
            if (
                move == temp_board.peek()
            ):  # Si el movimiento del bot es igual a un movimiento legal
                temp_move_legal = True
                break

        if temp_move_legal:
            illegal_move = False  # El movimiento es legal, así que salimos del bucle
        else:
            print("Movimiento ilegal, realice otro movimiento")
            # Actualiza los movimientos legales después de un movimiento ilegal

            legal_moves = list(board.legal_moves)

    # Obtiene parámetros globales de la partida
    fen = temp_board.fen()
    fen_parameteres = bot1.get_fen_parameters()
    turn = fen_parameteres["turn"]
    castling = fen_parameteres["castling"]
    en_passant = fen_parameteres["en_passant"]
    halfmove = fen_parameteres["halfmove"]
    fullmove = fen_parameteres["fullmove"]

    # Simula la detección
    board = fen_to_board(fen)
    fen = board_to_fen(board, turn, castling, en_passant, halfmove, fullmove)
    board = chess.Board(fen)
    print(board)

    bot2.set_board(board)
    temp_board = bot2.move()

    if temp_board is None:
        break

    # Obtiene parámetros globales de la partida
    fen = temp_board.fen()
    fen_parameteres = bot2.get_fen_parameters()
    turn = fen_parameteres["turn"]
    castling = fen_parameteres["castling"]
    en_passant = fen_parameteres["en_passant"]
    halfmove = fen_parameteres["halfmove"]
    fullmove = fen_parameteres["fullmove"]

# Game result
result = board.result()
if result == "1-0":
    print("Ganan las blancas.")
elif result == "0-1":
    print("Ganan las negras.")
else:
    print("Empate.")
