from config import *
import cv2
from enum import Enum, auto
import numpy as np
import time
from security_system import SecuritySystem, SecurityState
from picamera2 import Picamera2



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
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        while True:
            frame = self.capture_frame()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                self._state = SecurityState.STAGE1
                self._current_password = []
                print("Clearing sequence")
            if key == ord('s'):
                if self.check_if_correct():
                    print("Security system passed")
                    break
                else:
                    print("Incorrect Password")
                    self._state = SecurityState.STAGE1
                    self._current_password = []

            piece_detected = self.check_piece_in_frame(frame)

            if piece_detected in ["black", "white"] and self._earlier_piece != (
                piece_detected is not None
            ):
                
                self.stage_transition()
                self._current_password.append(piece_detected)
            frame = self.draw_on_frame(frame)
            cv2.imshow("Frame", frame)
            
                
            if piece_detected != "detected":
                self._earlier_piece = piece_detected is not None
        cv2.destroyAllWindows()
        self._cap.stop()


if __name__ == "__main__":
    security_system = RaspberrySecuritySystem()
    security_system.run()
