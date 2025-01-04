from config import *
import cv2
from enum import Enum, auto
import numpy as np
import time
from security_system import SecuritySystem
from picamera2 import Picamera2


class SecurityState(Enum):
    STAGE1 = auto()
    STAGE2 = auto()
    STAGE3 = auto()
    STAGE4 = auto()
    CORRECT_CODE = auto()


class RaspberrySecuritySystem(SecuritySystem):
    def __init__(self):
        super().__init__()
        self._cap = Picamera2()
        low_res_config = self._cap.create_still_configuration(
            main={"size": (1920, 1080)}
        )
        self._cap.configure(low_res_config)
        self._cap.start()

    def capture_frame(self):
        # Capture frame from camera
        # Initialize the camera

        # Check if the camera opened successfully
        frame = self._cap.capture_array()

        # Release the camera

        # Check if the frame was captured successfully
        if frame is not None:
            # Save the captured frame to a file
            return frame
        else:
            print("Error: Could not capture frame.")
            return None

    def run(self):
        print("Starting security system...")
        print("Press any key to take a photo")
        cv2.waitKey(0)
        frame = self.capture_frame()
        self.define_ROI(frame)
        print("You have to input the right security code to pass the security system")
        print("press c to clear the sequence")
        while self._state != SecurityState.CORRECT_CODE:

            frame = self.capture_frame()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                self._state = SecurityState.STAGE1
                print("Clearing sequence")

            piece_detected = self.check_piece_in_frame(frame)

            if piece_detected in ["black", "white"] and self._earlier_piece != (
                piece_detected is not None
            ):
                correct = self.check_if_correct(piece_detected)
                self.stage_transition(correct)
            frame = self.draw_on_frame(frame)
            cv2.imshow("Frame", frame)
            if self._state == SecurityState.CORRECT_CODE:
                print("Security system passed")
                break
            if piece_detected != "detected":
                self._earlier_piece = piece_detected is not None
        cv2.destroyAllWindows()
        self._cap.stop()


if __name__ == "__main__":
    security_system = SecuritySystem()
    security_system.run()
