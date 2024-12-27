from ChessBot import ChessBot
import pygame
import chess


def draw_board(board: chess.Board):
    for row in range(8):
        for col in range(8):
            color = (255, 255, 255) if (row + col) % 2 == 0 else (125, 135, 150)
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            piece = board.piece_at((7 - row) * 8 + col)
            if piece:
                piece_image = pieces_images.get(piece.symbol())
                screen.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()

    WIDTH, HEIGHT = 600, 600  # Size of the window
    SQUARE_SIZE = WIDTH // 8  # Size of each square

    # Load images for piecesç
    pieces_images = {
    "P": pygame.image.load("Piezas/wp.png"),
    "N": pygame.image.load("Piezas/wn.png"),
    "B": pygame.image.load("Piezas/wb.png"),
    "R": pygame.image.load("Piezas/wr.png"),
    "Q": pygame.image.load("Piezas/wq.png"),
    "K": pygame.image.load("Piezas/wk.png"),
    "p": pygame.image.load("Piezas/bp.png"),
    "n": pygame.image.load("Piezas/bn.png"),
    "b": pygame.image.load("Piezas/bb.png"),
    "r": pygame.image.load("Piezas/br.png"),
    "q": pygame.image.load("Piezas/bq.png"),
    "k": pygame.image.load("Piezas/bk.png")
    }

    for key, image in pieces_images.items():
        pieces_images[key] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))

    # Initialize the chess board
    board = chess.Board()
    stockfish_path = "stockfish-windows-x86-64-sse41-popcnt\stockfish\stockfish-windows-x86-64-sse41-popcnt.exe"
    bot1 = ChessBot(stockfish_path, depth=9)# , elo_rating=2500)
    bot2 = ChessBot(stockfish_path, depth=10)# , elo_rating=2500)

    # Set up the Pygame screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess Game")



    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        temp_board = bot1.move()
        if temp_board is None:
            break
        board = temp_board

        bot2.set_board(board)
        draw_board(board)
        pygame.display.flip()
        
        temp_board = bot2.move()
        if temp_board is None:
            break

        board = temp_board

        bot1.set_board(board)
        draw_board(board)
        pygame.display.flip()

    # Game result
    result = board.result()
    if result == "1-0":
        print("Ganan las blancas.")
    elif result == "0-1":
        print("Ganan las negras.")
    else:
        print("Empate.")