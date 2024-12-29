import cv2
import numpy as np
from copy import deepcopy
from GMM import read_video

# Lista para guardar las coordenadas de los puntos seleccionados
points = []

# Callback para manejar los clics del mouse
def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))  # Guardar las coordenadas del punto
        print(f"Punto seleccionado: {x}, {y}")
        # Dibujar el punto en la imagen para referencia visual
        cv2.circle(temp_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Selecciona las esquinas", temp_image)

def show_image(img: np.array, img_name: str = "Image", resize=False) -> None:
    """
    Display an image using OpenCV.

    Args:
        img (np.array): the image to display.
        img_name (str, optional): The name of the window. Defaults to "Image".
        resize (bool, optional): If the image is too big we can resize it. Defaults to False.
    """
    if resize:
        img = cv2.resize(img, (800, 600))
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Cargar el video y tomar el primer frame
videopath = "Video_Sergio_chess.mp4"
frames, frame_width, frame_height, frame_rate = read_video(videopath)
img = frames[0]


# Crear una copia temporal para dibujar los puntos seleccionados
temp_image = img.copy()

cv2.imshow("Selecciona las esquinas", temp_image)

cv2.setMouseCallback("Selecciona las esquinas", select_points)

print("Haz clic en las 4 esquinas del tablero de ajedrez en orden (sentido horario o antihorario). Presiona 'q' para continuar.")

# Esperar a que el usuario presione 'q' para cerrar
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()


# Ordenar los puntos en el orden correcto
points = np.array(points, dtype=np.float32)

# Perspectiva transformada para alinear el tablero (opcional)
width = 400  # Ancho deseado para la perspectiva
height = 400  # Alto deseado para la perspectiva
destination_points = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype=np.float32)

# Obtener la transformación de perspectiva
matrix = cv2.getPerspectiveTransform(points, destination_points)
warped = cv2.warpPerspective(img, matrix, (width, height))

# Convertir la imagen de la perspectiva ajustada a escala de grises
gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# Detección de esquinas en el tablero transformado
gray_blur = cv2.GaussianBlur(gray_warped, (5, 5), 3)
corners = cv2.goodFeaturesToTrack(
    gray_blur, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7
)

corners = np.int32(corners)

# Dibujar las esquinas detectadas en la imagen transformada
if corners is not None:
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(warped, (x, y), 5, (0, 255, 0), -1)

# Mostrar el tablero transformado con las esquinas detectadas
cv2.imshow("Tablero ajustado y esquinas detectadas", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()



winSize=(15, 15)
maxLevel=2
criteria= (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

mask = None
maxCorners = 100
qualityLevel = 0.3
minDistance = 7
blockSize = 7

prev_gray = gray_warped
p0 = corners.reshape(-1, 1, 2).astype(np.float32)  

# mask = np.zeros_like(warped)

for frame in frames[1::]: 
    input_frame = deepcopy(frame)
    warped_frame = cv2.warpPerspective(input_frame, matrix, (width, height))
    
    warped_frame_gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, warped_frame_gray, p0, None, winSize=winSize, maxLevel=maxLevel, criteria=criteria)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        warped_frame = cv2.circle(warped_frame, (a, b), 5, (0, 0, 255), -1)

    prev_gray = warped_frame_gray
    p0 = p1

    cv2.imshow('Frame', warped_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()


