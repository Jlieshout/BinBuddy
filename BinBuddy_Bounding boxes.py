# Importeren van de Modules
# Objectdetectie
from ultralytics import YOLO 
# Camera gebruik
import cv2
# Wiskundige berekeningen
import math

# Openen webcam
cap = cv2.VideoCapture(0)
# Breedte van webcamscherm
cap.set(3, 640)
# Hoogte van webcamscherm
cap.set(4, 480)

# Pad naar het yolo bestand dat wordt ingeladen
model = YOLO("/Users/jlavanlieshout/Desktop/best.pt")

# De objecten die het model kan herkennen + label namen
classNames = ["GFT" , "Papier" , "Plastic" , "Restafval" , "Statiegeld"
              ]

# Het continu vastleggen van webcamframes
while True:
    success, img = cap.read()
    # Het vastgelegde frame wordt doorgegeven aan het YOLO model en het model retouneert de resultaten
    results = model(img, stream=True)

    # Ophalen van de bounding boxes
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Coördinaten ban de bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            # Coördinaten omzetten naar gehele getallen
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Het tekenen van de bounding box op het scherm
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Berekend de betrouwbaarheid van de detectie
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # Bekijkt de classe index van het gedetecteerde object en geeft de naam van het object weer
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # Het toevoegen van de tekst op het scherm
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

# Het webcam scherm weergeven in een apart venster
    cv2.imshow('BinBuddy', img)
    # Het stoppen van de webcam loop wanneer 'q' wordt ingedrukt
    if cv2.waitKey(1) == ord('q'):
        break
    
# Hiermee wordt de webcam afgesloten en sluit het venster
cap.release()
cv2.destroyAllWindows()