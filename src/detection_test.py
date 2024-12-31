from KnightSight import KnightSight
import cv2
from picamera2 import Picamera2


if __name__ == "__main__":
    print("Starting KnightSight...")
    knight_sight = KnightSight()
    # Initialize the camera
    print("Place the camera pointing to the chessboard")
    cv2.waitKey(0)
    picam2 = Picamera2()

    # Configure the camera
    picam2.configure(picam2.create_still_configuration())

    # Start the camera
    picam2.start()

    first_frame = picam2.capture_array()

    # Stop the camera
    picam2.stop()

    knight_sight.initialise_first_frame(first_frame)

    print("KnightSight has been set up.")
    print("Press 'q' to quit")
    cap = cv2.VideoCapture(0)
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break
        knight_sight.process_frame(frame)
