import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import mysql.connector
from datetime import datetime

model = YOLO('yolov8n.pt')

area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Bernardito1",
    database="control_acceso"
)

cursor = db.cursor()

cap = cv2.VideoCapture('peoplecount1.mp4')


frame_width = 1020
frame_height = 500
fps = int(cap.get(cv2.CAP_PROP_FPS))


out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))


with open("coco.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")


tracker = Tracker()
people_entering = {}
people_exiting = {}
entering = set()
exiting = set()

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (frame_width, frame_height))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    object_list = []

    for index, row in px.iterrows():
        x1, y1, x2, y2 = map(int, [row[0], row[1], row[2], row[3]])
        class_id = int(row[5])
        class_name = class_list[class_id]

        if 'person' in class_name:
            object_list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(object_list)

    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox
        if cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False) >= 0:
            people_entering[obj_id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        if obj_id in people_entering:
            if cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False) >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(obj_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                if obj_id not in entering:
                    entering.add(obj_id)
                    cursor.execute("INSERT INTO registros (type) VALUES ('entrada')")
                    db.commit()

        if cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False) >= 0:
            people_exiting[obj_id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        if obj_id in people_exiting:
            if cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False) >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(obj_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                if obj_id not in exiting:
                    exiting.add(obj_id)
                    cursor.execute("INSERT INTO registros (type) VALUES ('salida')")
                    db.commit()

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, '1', (504, 471), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, '2', (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    
    cv2.putText(frame, f"Entradas: {len(entering)}", (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Salidas: {len(exiting)}", (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    
    
    num_personas = len(object_list)
    text = f"Personas: {num_personas}"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)
    x_pos = 770
    y_pos = 70   
    cv2.rectangle(frame, (x_pos, y_pos - text_height - baseline), (x_pos + text_width, y_pos + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    # Guardar el frame en el video de salida
    out.write(frame)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cursor.close()
db.close()


out.release()
cap.release()
cv2.destroyAllWindows()
