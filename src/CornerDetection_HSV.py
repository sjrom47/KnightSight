import cv2
import numpy as np
from glob import glob

def show_image(img, img_name = "img", resize = False):
    if resize:
        img = cv2.resize(img, (500, 500))

    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_images(filenames, resize = True):
    imgs = [cv2.imread(filename) for filename in filenames]

    if resize:
        return [cv2.resize(img, (500, 500)) for img in imgs]
    
    return imgs
    
def nothing(x):
    pass

def get_hsv_color_ranges(image: np.array):

    # Create a window
    cv2.namedWindow('image')

    # Create trackbars for color change
    cv2.createTrackbar('HMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 255)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    output = image
    wait_time = 33

    while(1):

        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')

        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image,image, mask= mask)

        # Print if there is a change in HSV value
        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        cv2.imshow('image',output)
        cv2.resizeWindow("image", 500,300)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":

    imgs_path = glob("photos/*.jpg")
    imgs = load_images(imgs_path)

    print(imgs[3].shape)
    # get_hsv_color_ranges(imgs[3])

    lower_blue = np.array([86, 98, 95])
    upper_blue = np.array([170, 255, 255])

    hsv = cv2.cvtColor(imgs[3], cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_blue_inv = cv2.bitwise_not(mask_blue)
    # show_image(mask_blue_inv, "mask_blue_inv")


    found, corners = cv2.findChessboardCorners(mask_blue_inv, (7, 7))

    if found: 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners_refined = cv2.cornerSubPix(mask_blue_inv, corners, (5, 5), (-1, -1), criteria)
        print(corners_refined)

        corners_drawn = cv2.drawChessboardCorners(imgs[3], (7, 7), corners, True)
        corners_drawn_refined = cv2.drawChessboardCorners(imgs[3], (7, 7), corners_refined, True)

        show_image(corners_drawn, "Corners Drawn")
        show_image(corners_drawn_refined, "Corners Drawn")

        

