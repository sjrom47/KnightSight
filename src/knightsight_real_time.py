from KnightSight import KnightSight
import cv2
from picamera2 import Picamera2


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

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("frames", frame)
        if key == 13:  # Enter key
            knight_sight.process_frame(frame)
    picam2.stop()
    cv2.destroyAllWindows()
