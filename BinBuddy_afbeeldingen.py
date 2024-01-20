# Importeren van de Modules die nodig zijn
from ultralytics import YOLO #
import cv2                      #voor het gebruik van computer vision en beeld- videobewerking
import math                     #voor wiskundige berekeningen
import time                     #gebruik van tijd

# Toegang tot de webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)      #hoogte van webcamscherm
cap.set(4, 480)      #breedte van webcamscherm

# Pad naar het eigen YOLO-model dat wordt ingeladen
model = YOLO("/Users/jlavanlieshout/Desktop/best.pt")

# Variabele labelen van de lijst met objecten die het model kan herkennen
classNames = ["GFT", "Papier", "Plastic", "Restafval", "Statiegeld"]

# Inladen van de Afbeeldingen voor elke afvalcategorie
images = {
    "GFT": cv2.imread("/Users/jlavanlieshout/Desktop/GFT.png"),
    "Papier": cv2.imread("/Users/jlavanlieshout/Desktop/Papier.png"),
    "Plastic": cv2.imread("/Users/jlavanlieshout/Desktop/Plastic.png"),
    "Restafval": cv2.imread("//Users/jlavanlieshout/Desktop/Rest.png"),
    "Statiegeld": cv2.imread("/Users/jlavanlieshout/Desktop/Statiegeld.png"),
}

# Variabele om bij te houden of er al iets is gedetecteerd (ingesteld op geen detectie)
detectie_gevonden = False

# Het continu vastleggen van webcamframes (blijft in een loop zolang het succesvol frames vastlegt)
while True:
    success, img = cap.read()

    if not detectie_gevonden:
        # Het vastgelegde frame wordt doorgegeven aan het YOLO-model en het model retourneert de resultaten
        results = model(img, stream=True)

        # Ophalen van de bounding boxes. Omlijnen van het gevonden object
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Coördinaten van de bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                # Coördinaten omzetten naar gehele getallen (nodig voor de pixelcoördinatie)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Het tekenen van de bounding boxes
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Berekend de betrouwbaarheid van de detectie
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # Bekijkt welk object is gedetecteerd en geeft de naam van het object weer
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # Het weergeven van de afbeelding op het scherm
                if classNames[cls] in images:
                   img_h, img_w, _ = img.shape      # Ophalen van afmeting van huidige webcambeeld
                   img_resized = cv2.resize(images[classNames[cls]], (int(img_w), int(img_h)))      # De afbeelding wordt aangepast aan het formaat van het webcambeeld
                   img[0:img_resized.shape[0], 0:img_resized.shape[1]] = img_resized        # De afbeelding vervangt het webcam beeld

                # Het webcam scherm weergeven in een apart venster
                cv2.imshow('BinBuddy', img)
                cv2.waitKey(2000)       #Beeld tijdelijk stoppen voordat verdere actie plaats vindt

                # Wanneer een object is gedetecteerd wordt er vertelt waar het moet worden weggegooid
                if classNames[cls] in ["GFT"]:
                    detectie_gevonden = True
                    print("GFT, plaats in bak C.")
                if classNames[cls] in ["Plastic"]:
                    detectie_gevonden = True
                    print("Plastic, plaats in bak B.")
                if classNames[cls] in ["Restafval"]:
                    detectie_gevonden = True
                    print("Restafval, plaats in bak A.")
                if classNames[cls] in ["Papier"]:
                    detectie_gevonden = True
                    print("Papier, gooi weg bij Papier.")
                if classNames[cls] in ["Statiegeld"]:
                    detectie_gevonden = True
                    print("Statiegeld, breng naar emballage-automaat.")
          
                # Wanneer er een detectie is gevonden wordt de code twee minten gepauzeerd
                if detectie_gevonden:
                    time.sleep(2)
                    detectie_gevonden = False  #Reset de detectiestatus om opnieuw te beginnen
                   
        # Het webcam scherm weergeven in een apart venster
        cv2.imshow('BinBuddy', img)

    # Het stoppen van de webcam loop wanneer 'q' wordt ingedrukt
    if cv2.waitKey(1) == ord('q'):
        break

# Hiermee wordt de webcam afgesloten en sluit het venster
cap.release()
cv2.destroyAllWindows()
