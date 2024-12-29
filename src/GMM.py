import os
import cv2


def read_video(videopath: str):    
    """
    Reads a video file and returns its frames along with video properties.

    Args:
        videopath (str): The path to the video file.

    Returns:
        tuple: A tuple containing:
            - frames (list): A list of frames read from the video.
            - frame_width (int): The width of the video frames.
            - frame_height (int): The height of the video frames.
            - frame_rate (float): The frame rate of the video.
    """

    # Complete this line to read the video file
    cap = cv2.VideoCapture(videopath) 
    
    # Check if the video was successfully opened
    if not cap.isOpened():
        print('Error: Could not open the video file')
        return 

    # Get the szie of frames and the frame rate of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Get the width of the video frames
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get the height of the video frames
    frame_rate = cap.get(cv2.CAP_PROP_FPS) # Get the frame rate of the video
    
    # Use a loop to read the frames of the video and store them in a list
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, frame_width, frame_height, frame_rate



videopath = "Video_Sergio_chess.mp4"
frames, frame_width, frame_height, frame_rate = read_video(videopath)

history = 200  # Number of frames to use to build the background model
varThreshold = 25  # Threshold to detect the background
detectShadows = False  # If True the algorithm detects the shadows

# Create the MOG2 object
mog2 = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)


folder_path = 'output_gmm' 
videoname = f'output_{history}_{varThreshold}_{detectShadows}.avi' 


# Create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec to use
frame_size = (frame_width, frame_height) # Size of the frames
fps = frame_rate # Frame rate of the video
out = cv2.VideoWriter(os.path.join(folder_path, videoname), fourcc, fps, frame_size)

for frame in frames:
    # Apply the MOG2 algorithm to detect the moving objects
    mask = mog2.apply(frame)

    foreground_pixels = cv2.countNonZero(mask)
    total_pixels = mask.size
    

    # Convert to BGR the mask to store it in the video
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Save the mask in a video
    out.write(mask)

out.release()


