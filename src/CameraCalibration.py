import numpy as np
import copy
import glob
import cv2
from utils import get_hsv_color_ranges

def read_resize_images(path, scale_factor):
    """
    Lee y redimensiona las imágenes de un directorio.
    
    Parameters:
    path (str): Ruta del directorio que contiene las imágenes.
    scale_factor (float): Factor de escala para redimensionar las imágenes.
    
    Returns:
    list: Lista de imágenes en formato BGR.
    """
    imgs = [cv2.imread(i) for i in glob.glob(path)]
    return [cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor) for img in imgs]

def get_chessboard_points(chessboard_shape, dx, dy):
    """
    Genera los puntos 3D del patrón del tablero de ajedrez en el espacio del mundo.
    
    Parameters:
    chessboard_shape (tuple): Dimensiones del tablero de ajedrez (filas, columnas).
    dx (float): Tamaño del paso en el eje X (distancia entre casillas horizontalmente).
    dy (float): Tamaño del paso en el eje Y (distancia entre casillas verticalmente).
    
    Returns:
    np.ndarray: Array de puntos 3D en el espacio del mundo con coordenadas Z igual a 0.
    """
    return np.array(
        [[i * dx, j * dy, 0] for i in range(min(chessboard_shape)) for j in range(max(chessboard_shape))],
        dtype=np.float32
    )

def calibrate_camera(imgs, dim): 
    """
    Calibra una cámara usando imágenes de un tablero de ajedrez.
    
    Parameters:
    imgs (list): Lista de imágenes en formato BGR que contienen el patrón del tablero de ajedrez.
    dim (tuple): Dimensiones del tablero de ajedrez (filas, columnas) utilizado para la calibración.
    
    Returns:
    tuple: Contiene los siguientes valores:
        - rms (float): Error medio cuadrático (Root Mean Square) de la calibración.
        - intrinsics (np.ndarray): Matriz intrínseca de la cámara.
        - extrinsics (list): Lista de matrices extrínsecas para cada imagen.
        - dist_coeffs (np.ndarray): Coeficientes de distorsión de la cámara.
    """
    # Detecta las esquinas del tablero de ajedrez en las imágenes
    corners = [cv2.findChessboardCorners(img, dim) for img in imgs]
    corners_copy = copy.deepcopy(corners)  # Copia de las esquinas para refinarlas más adelante

    # Criterio de terminación para el refinamiento de las esquinas
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    
    # Convierte las imágenes a escala de grises
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    
    # Refinamiento de las esquinas detectadas usando cornerSubPix
    corners_refined = [
        cv2.cornerSubPix(img_gray, cor[1], (11, 11), (-1, -1), criteria) 
        for img_gray, cor in zip(imgs_gray, corners_copy) 
        if cor[0]  # Solo refina si se encontraron esquinas
    ]
    
    # Genera los puntos 3D del tablero de ajedrez para cada imagen
    chessboard_points = [get_chessboard_points(dim, 30, 30) for cor in corners_refined]
    
    # Calibración de la cámara
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        chessboard_points, 
        corners_refined, 
        imgs[0].shape[:-1],  # Tamaño de la imagen (alto, ancho)
        None, 
        None
    )
    
    # Cálculo de las matrices extrínsecas (rotación y traslación)
    extrinsics = [
        np.hstack((cv2.Rodrigues(rvec)[0], tvec)) 
        for rvec, tvec in zip(rvecs, tvecs)
    ]
    
    # Devuelve los resultados de la calibración
    return rms, intrinsics, extrinsics, dist_coeffs

def calibrate_camera_2(imgs, dim, visualize=False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    lower_blue = np.array([86, 98, 95])
    upper_blue = np.array([170, 255, 255])

    hsv_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in imgs]
    
    masks_blue = [cv2.inRange(hsv, lower_blue, upper_blue) for hsv in hsv_imgs]
    masks_blue_inv = [cv2.bitwise_not(mask_blue) for mask_blue in masks_blue]

    corners = [cv2.findChessboardCorners(mask_blue_inv, dim) for mask_blue_inv in masks_blue_inv]
    corners_refined = [cv2.cornerSubPix(mask_blue_inv, cor[1], (5, 5), (-1, -1), criteria) for mask_blue_inv, cor in zip(masks_blue_inv, corners)]

    chessboard_points = [get_chessboard_points(dim, 30, 30) for cor in corners_refined]
    
    # Calibración de la cámara
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        chessboard_points, 
        corners_refined, 
        imgs[0].shape[:-1],  # Tamaño de la imagen (alto, ancho)
        None, 
        None
    )
    
    # Cálculo de las matrices extrínsecas (rotación y traslación)
    extrinsics = [
        np.hstack((cv2.Rodrigues(rvec)[0], tvec)) 
        for rvec, tvec in zip(rvecs, tvecs)
    ]

    if visualize:
        for img, mask_blue_inv, cor, cor_refined in zip(imgs, masks_blue_inv, corners, corners_refined):
            img_drawn = cv2.drawChessboardCorners(img, dim, cor[1], True)
            img_drawn_refined = cv2.drawChessboardCorners(img, dim, cor_refined, True)
            cv2.imshow('img', img_drawn)
            cv2.imshow('img', img_drawn_refined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Devuelve los resultados de la calibración
    return rms, intrinsics, extrinsics, dist_coeffs

if __name__ == "__main__": 
    
    def show_image(img):
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    lower_blue = np.array([86, 98, 95])
    upper_blue = np.array([170, 255, 255])

    dim = (7, 7)

    imgs = read_resize_images("calibration_photos/*.jpg", 0.2)
    
    # rms, intrinsics, extrinsics, dist_coeffs = calibrate_camera(imgs, dim)
    # print(rms)

    rms, intrinsics, extrinsics, dist_coeffs = calibrate_camera_2(imgs, dim)
    print(rms)
    print(dist_coeffs)
    print(intrinsics)

    # Imagen de ejemplo
    img = imgs[0]

    undistorted_img = cv2.undistort(img, intrinsics, dist_coeffs)

    # Muestra las imágenes antes y después
    cv2.imshow('Original', img)
    cv2.imshow('Undistorted', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

    