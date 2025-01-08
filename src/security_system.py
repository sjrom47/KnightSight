from config import *
import cv2
from enum import Enum, auto
import numpy as np
import time


class SecurityState(Enum):
    STAGE1 = auto()
    STAGE2 = auto()
    STAGE3 = auto()
    STAGE4 = auto()
    CORRECT_CODE = auto()


class SecuritySystem:
    def __init__(self):
        self._state = SecurityState.STAGE1
        self._track_window = None
        self._white_mask = (
            PIECE_COLOR_MASKS["white"][0],
            PIECE_COLOR_MASKS["white"][1],
        )
        self._black_mask = (
            PIECE_COLOR_MASKS["black"][0],
            PIECE_COLOR_MASKS["black"][1],
        )
        self._earlier_piece = False
        self._cap = cv2.VideoCapture(0)
        self._current_password = []

    def capture_frame(self):
        # Capture frame from camera
        # Initialize the camera

        # Check if the camera opened successfully
        if not self._cap.isOpened():
            print("Error: Could not open camera.")
            exit()

        # Capture a single frame
        ret, frame = self._cap.read()

        # Release the camera

        # Check if the frame was captured successfully
        if ret:
            # Save the captured frame to a file
            return frame
        else:
            print("Error: Could not capture frame.")
            return None

    def define_ROI(self, frame):
        x, y, w, h = cv2.selectROI("Frame", frame, False)
        self._track_window = (x, y, w, h)

    def stage_transition(self):
        if self._state == SecurityState.STAGE1:
            self._state = SecurityState.STAGE2
            print("Stage 1")
        elif self._state == SecurityState.STAGE2:
            self._state = SecurityState.STAGE3
            print("Stage 2")
        elif self._state == SecurityState.STAGE3:
            self._state = SecurityState.STAGE4
            print("Stage 3")
        elif self._state == SecurityState.STAGE4:
            self._state = SecurityState.CORRECT_CODE
            print("Stage 4")

        elif self._state == SecurityState.CORRECT_CODE:
            self._state = SecurityState.CORRECT_CODE
        else:
            raise ValueError("Invalid state")

    def check_if_correct(self):
        return self._current_password == SECURITY_CODE

    def check_piece_in_frame(self, frame):
        x, y, w, h = self._track_window
        roi = frame[y : y + h, x : x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_presence = cv2.inRange(hsv_roi, *self._white_mask)

        black_presence = cv2.inRange(hsv_roi, *self._black_mask)
        white_count = cv2.countNonZero(white_presence)
        black_count = cv2.countNonZero(black_presence)

        circles = cv2.HoughCircles(
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=50,
            param2=30,
            minRadius=0,
            maxRadius=0,
        )
        
        if max(white_count, black_count) > 200 and circles is not None:
            print("Piece detected")
            if white_count > black_count:
                print("White piece detected")
                piece_detected = "white"
            else:
                print("Black piece detected")
                piece_detected = "black"
            return piece_detected
        elif max(white_count, black_count) > 200:
            return "detected"

    def draw_on_frame(self, frame):
        x, y, w, h = self._track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for i in range(self._state.value - 1):
            mean_hsv = np.uint8(np.mean(PIECE_COLOR_MASKS[self._current_password[i]], axis=0))
            mean_hsv = mean_hsv.reshape(1, 1, 3)
            # Convert to BGR
            mean_bgr = cv2.cvtColor(mean_hsv, cv2.COLOR_HSV2BGR)
            # Get the actual BGR values (removes the extra dimensions)
            mean_bgr = mean_bgr[0][0]
            cv2.rectangle(
                frame,
                (10 + 25 * i, 10),
                (30 + 25 * i, 30),
                mean_bgr.tolist(),
                -1,
            )
        return frame

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
        self._cap.release()


if __name__ == "__main__":
    security_system = SecuritySystem()
    security_system.run()
