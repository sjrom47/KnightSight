from picamera2 import Picamera2
from CameraCalibration import calibrate_camera
import cv2
import time
import glob

# Initialize the camera
picam2 = Picamera2()

# Configure the camera
picam2.configure(picam2.create_still_configuration())

# Start the camera
picam2.start()

# Capture 5 images
for i in range(3):
    filename = f"data/photos/extra/image_{i+1}.jpg"
    picam2.capture_file(filename)
    print(f"Captured {filename}")
    time.sleep(3)  # Wait for 1 second between captures

# Stop the camera
picam2.stop()

# dim = (7, 7)

# imgs_path = [i for i in glob.glob("../data/calibration_photos/image_*.jpg")]
# imgs = [cv2.imread(i) for i in imgs_path]
# print(imgs)

# rms, intrinsics, extrinsics, dist_coeffs = calibrate_camera(imgs, dim)
# print(rms)
