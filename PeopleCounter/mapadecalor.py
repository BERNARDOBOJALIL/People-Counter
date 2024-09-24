import cv2
import numpy as np
from ultralytics import YOLO

# Cargar el modelo YOLOv8 con segmentación
model = YOLO("yolov8n-seg.pt")  # Asegúrate de usar el modelo con segmentación

# Cargar el video
video_path = "videoplayback.mp4"
cap = cv2.VideoCapture(video_path)

# Leer el primer frame para obtener las dimensiones
ret, frame = cap.read()
heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

# Parámetros
alpha = 0.6
cooling_rate = 0.05

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detección con segmentación
    results = model(frame)[0]
    
    # Crear una máscara global para todas las personas detectadas
    detection_mask = np.zeros_like(heatmap, dtype=np.float32)
    
    if results.masks is not None:  # Verificar si hay máscaras
        masks = results.masks.data.cpu().numpy()  # Convertir las máscaras a un arreglo NumPy
        
        for mask in masks:
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Redimensionar la máscara al tamaño del frame
            detection_mask += mask_resized.astype(np.float32)  # Añadir la máscara al mapa de detección

    # Aplicar enfriamiento selectivo (solo en áreas donde detection_mask es 0)
    heatmap[detection_mask == 0] = np.maximum(heatmap[detection_mask == 0] - cooling_rate, 0)
    
    # Añadir nuevas detecciones al mapa de calor
    heatmap += detection_mask

    # Normalizar el mapa de calor a rango 0-255
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convertir a mapa de calor en color
    heatmap_color = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    
    # Superponer el mapa de calor sobre el frame original
    overlay = cv2.addWeighted(heatmap_color, alpha, frame, 1 - alpha, 0)

    # Mostrar el resultado
    cv2.imshow('Heatmap', overlay)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

