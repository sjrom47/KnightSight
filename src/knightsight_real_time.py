from KnightSightv2 import KnightSight
import cv2
from picamera2 import Picamera2
import pygame

def draw_board(board):
    for row in range(8):
        for col in range(8):
            color = (255, 255, 255) if (row + col) % 2 == 0 else (125, 135, 150)
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            piece = board.state[row][col]
            if piece:
                piece_image = pieces_images.get(piece)
                screen.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))


if __name__ == "__main__":
    print("Starting KnightSight...")
    knight_sight = KnightSight()
    # Initialize the camera
    print("Place the camera pointing to the chessboard")

    picam2 = Picamera2()

    full_res_config = picam2.create_still_configuration(
        main={"size": picam2.sensor_resolution}
    )
    picam2.configure(full_res_config)
    picam2.start()
    first_frame = picam2.capture_array()

    # Stop the camera
    picam2.stop()

    knight_sight.initialise_first_frame(first_frame, override=True)

    print("KnightSight has been set up.")
    print("Press 'q' to quit")
    low_res_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
    picam2.configure(low_res_config)
    picam2.start()
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
    -6: pygame.image.load("data/Piezas/bk.png")
    }

    for key, image in pieces_images.items():
        pieces_images[key] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))

    # Set up the Pygame screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess Game")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("frames", frame_rgb)
        if key == 13:  # Enter key
            knight_sight.process_frame(frame_rgb)
        draw_board(knight_sight.visual_board)
        pygame.display.flip()
    picam2.stop()
    cv2.destroyAllWindows()


    
    



    
        


        
        

    