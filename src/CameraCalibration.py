import numpy as np
import copy
import glob
import cv2
import os


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
        [
            [i * dx, j * dy, 0]
            for i in range(min(chessboard_shape))
            for j in range(max(chessboard_shape))
        ],
        dtype=np.float32,
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
    imgs_copy = copy.deepcopy(
        imgs
    )  # Copia de las imágenes para dibujar las esquinas detectadas
    # Detecta las esquinas del tablero de ajedrez en las imágenes
    corners = [cv2.findChessboardCorners(img, (8, 6)) for img in imgs]
    corners_copy = copy.deepcopy(
        corners
    )  # Copia de las esquinas para refinarlas más adelante

    # Criterio de terminación para el refinamiento de las esquinas
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    # Convierte las imágenes a escala de grises
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    # Refinamiento de las esquinas detectadas usando cornerSubPix
    corners_refined = [
        cv2.cornerSubPix(img_gray, cor[1], (5, 5), (-1, -1), criteria)
        for img_gray, cor in zip(imgs_gray, corners_copy)
        if cor[0]  # Solo refina si se encontraron esquinas
    ]
    path = "./data/"
    for img, cor, img_path in zip(imgs_copy, corners_refined, imgs_path):
        var = False if len(cor) == 0 else True
        if var:
            cv2.drawChessboardCorners(img, (8, 6), cor, var)
            show_image(img)
            write_image(img, path, img_path)

    # Genera los puntos 3D del tablero de ajedrez para cada imagen
    chessboard_points = [get_chessboard_points(dim, 30, 30) for cor in corners_refined]

    # Calibración de la cámara
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        chessboard_points,
        corners_refined,
        imgs[0].shape[:-1],  # Tamaño de la imagen (alto, ancho)
        None,
        None,
    )

    # Cálculo de las matrices extrínsecas (rotación y traslación)
    extrinsics = [
        np.hstack((cv2.Rodrigues(rvec)[0], tvec)) for rvec, tvec in zip(rvecs, tvecs)
    ]

    # Devuelve los resultados de la calibración
    return rms, intrinsics, extrinsics, dist_coeffs


if __name__ == "__main__":

    def show_image(img):
        cv2.imshow("img", img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    def write_image(img, path, img_path, img_type="cal"):
        os.makedirs(path, exist_ok=True)
        image_name = img_path.rsplit("\\")[-1].rstrip(".jpg")
        cv2.imwrite(f"{path}/{image_name}_{img_type}.jpg", img)

    dim = (8, 6)

    imgs_path = [i for i in glob.glob("./data/calibration_photos/*.jpg")]
    imgs = [cv2.imread(i) for i in imgs_path]

    rms, intrinsics, extrinsics, dist_coeffs = calibrate_camera(imgs, dim)
    print(rms)

    imgs_path = [i for i in glob.glob("right/*.jpg")]
    imgs = [cv2.imread(i) for i in imgs_path]

    rms, intrinsics, extrinsics, dist_coeffs = calibrate_camera(imgs, dim)
    print(rms)
