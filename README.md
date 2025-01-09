# Knightsight: Computer Vision Chess Movement Detection 🎥♟️

Knightsight is a computer vision project developed to detect and analyze chess movements in real-time, created as part of the Computer Vision course.

<!-- TABLE OF CONTENTS -->
- [Knightsight: Computer Vision Chess Movement Detection 🎥♟️](#knightsight-computer-vision-chess-movement-detection-️)
  - [About the Project](#about-the-project)
  - [Key Features ✨](#key-features-)
    - [Camera Calibration](#camera-calibration)
    - [Chess Movement Detection](#chess-movement-detection)
    - [Security System](#security-system)
    - [Real-time Processing](#real-time-processing)
  - [System Architecture 🔧](#system-architecture-)
  - [Libraries and Dependencies 📚](#libraries-and-dependencies-)
  - [Running the Application 🚀](#running-the-application-)
  - [Future Developments 🔮](#future-developments-)
  - [Developers 👨‍💻](#developers-)

## About the Project
**Knightsight** is a computer vision system designed to detect and analyze chess movements through video capture. The project combines various image processing techniques with chess game validation to create an interactive and educational tool for chess players.

## Key Features ✨

### Camera Calibration
- **Custom Chessboard Detection:** Utilizes a blue and yellow chessboard pattern with mask adaptation
- **Distortion Correction:** Calculates and applies camera parameters for accurate image processing

### Chess Movement Detection
- **Board Corner Detection:** Uses Delaunay triangulation with RANSAC algorithm
- **Hand Movement Tracking:** Implements Gaussian Mixture Model (GMM) for hand detection
- **Corner Tracking:** Using the Lucas Kanade Optical flow algorithm
- **Piece Recognition:** Employs "bag of words" technique for chess piece classification. Includes a program to label pieces from photos to create a training set and a program to perform cross validation
- **Move Validation:** Integrates with Stockfish chess engine for legal move verification

### Security System
- **Password System:** Uses chess pawns as input elements
- **Pattern Recognition:** Implements Hough transform for circular pattern detection
- **Error Handling:** Includes reset functionality for incorrect entries

### Real-time Processing
- **Dual Versions:**
  - Real-time version optimized for Raspberry Pi that requires manual intervention
  - Full-featured version for video analysis
- **Visual Feedback:** Displays board corners and hand detection in real-time
- **Move Visualization:** Uses pygame for game state representation

## System Architecture 🔧
The system follows a modular approach with several key components:
- Image acquisition and preprocessing
- Board and piece detection
- Hand tracking and movement analysis
- Game state validation and visualization

## Libraries and Dependencies 📚
> Python 3.x recommended for optimal compatibility

Essential libraries include:
```bash
opencv-python
numpy
pygame
stockfish
```
They are included in the requirements.txt file.

There are 2 additional dependencies:
1. Picamera2: if you want to use this project in a Raspberry Pi we strongly advise using this library. Set it up and create the environment with `--system-site-packages`
  
2. Stockfish: you will need to download the appropriate version from the official pages

## Running the Application 🚀
1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the application:
```bash
python src/Knightsight.py
```

3. System Usage:
   * Complete the security system verification
   * Position the camera above the chessboard
   * Begin playing chess - the system will automatically detect and validate moves (not advised for weaker systems. In this case you can load a video and let it process)
  
This is a demo on a Raspberry Pi 5:

https://github.com/user-attachments/assets/4a043cff-a562-4882-812d-d6377463e459



## Future Developments 🔮
- AI opponent integration with adjustable difficulty using Stockfish
- Post-game analysis mode
- Advanced gesture recognition
- Online multiplayer support
- Enhanced illumination normalization
- Improved camera movement tolerance
- Improved performance on less powerful systems

## Developers 👨‍💻
- [Sergio Jiménez Romero](https://github.com/sjrom47)
- [Carlos Martínez Cuenca](https://github.com/carlosIMAT)
