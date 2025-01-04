from utils import load_video, get_hsv_color_ranges
import numpy as np
import copy
import cv2


class KalmanFilter:
    def __init__(self, min_range_hsv, max_range_hsv):
        self.min_range_hsv = min_range_hsv
        self.max_range_hsv = max_range_hsv

        # Inicialización del filtro de Kalman
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-5

        self.track_window = None
        self.i = None
        self.crop_hist = None

    def initialize(self, frames):
        for i, frame in enumerate(frames):
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(0)
            if key == ord("n"):
                continue
            elif key == ord("s"):
                x, y, w, h = cv2.selectROI("Frame", frame, False)
                self.track_window = (x, y, w, h)
                cx = int(x + w / 2)
                cy = int(y + h / 2)

                # Inicialización del estado posterior
                self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
                self.kf.errorCovPost = np.eye(4, dtype=np.float32)

                # Histogram calculation
                crop = frame[y : y + h, x : x + w].copy()
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_crop, self.min_range_hsv, self.max_range_hsv)
                crop_hist = cv2.calcHist([hsv_crop], [0], mask, [180], [0, 180])
                cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)

                self.i = i
                self.crop_hist = crop_hist
                break

        cv2.destroyAllWindows()

    def predict(self, frames, visualize=True):
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)
        frames_predictions = []
        predictions = []

        for frame in frames[self.i :]:
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            img_bproject = cv2.calcBackProject(
                [img_hsv], [0], self.crop_hist, [0, 180], 1
            )

            ret, self.track_window = cv2.meanShift(
                img_bproject, self.track_window, term_crit
            )
            x_, y_, w_, h_ = self.track_window
            c_x, c_y = x_ + w_ // 2, y_ + h_ // 2

            # Predicción y corrección
            prediction = self.kf.predict()
            measurement = np.array([[c_x], [c_y]], np.float32)
            self.kf.correct(measurement)

            frames_predictions.append(frame)
            predictions.append(prediction)
            # Dibujar resultados
            if visualize:
                input_frame = frame.copy()
                cv2.circle(
                    input_frame,
                    (int(prediction[0]), int(prediction[1])),
                    5,
                    (0, 0, 255),
                    -1,
                )
                cv2.circle(input_frame, (c_x, c_y), 5, (0, 255, 0), -1)
                cv2.rectangle(input_frame, (x_, y_), (x_ + w_, y_ + h_), (255, 0, 0), 2)
                cv2.imshow("Frame", input_frame)

                if cv2.waitKey(2) == ord("q"):
                    break

        cv2.destroyAllWindows()

        return frames, predictions


if __name__ == "__main__":
    videopath = "data/other_data/videos/video_test_1.mp4"
    frames = load_video(videopath)

    # This ranges are created for the hand of the player
    min_range = (0, 63, 131)
    max_range = (255, 255, 255)

    kf = KalmanFilter(min_range, max_range)
    kf.initialize(frames)
    frames, predictions = kf.predict(frames)
