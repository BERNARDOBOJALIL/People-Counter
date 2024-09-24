import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("yolov8n-seg.pt") 


video_path = "videoplayback.mp4"
cap = cv2.VideoCapture(video_path)


ret, frame = cap.read()
heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

alpha = 0.6
cooling_rate = 0.05

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    if results.masks is not None:  
        masks = results.masks.data.cpu().numpy()  
        
        for mask in masks:
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  
            heatmap += mask_resized.astype(np.float32)  

    heatmap = np.maximum(heatmap - cooling_rate, 0)
    
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    

    heatmap_color = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(heatmap_color, alpha, frame, 1 - alpha, 0)

    cv2.imshow('Heatmap', overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
