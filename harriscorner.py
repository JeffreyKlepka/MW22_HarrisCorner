import cv2 
import numpy as np
from matplotlib import pyplot as plt

# Bild einlesen
image = cv2.imread('StreetInNamibia2.jpg') 
# Das Bild in Graustufen konvertieren  
operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
# Datentyp modifizieren. auf 32bit floating point setzen.  
operatedImage = np.float32(operatedImage) 
# den Algorithmus auf das Bild anwenden 
dest = cv2.cornerHarris(operatedImage, 2, 3, 0.01) 
dest = cv2.dilate(dest, None) 
# Das Bild in seine Ursprungsdarstellung überführen 
# Grenzwert festlegen und Punkte markieren
thresh=0.01 * dest.max()
image[dest > thresh]=[0, 0, 255] 
# Das Bild mit detektierten Ecken anzeigen 
plt.figure("Harris-Detector")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Harris-Detektor - Beispiel 2")
plt.show()
# Speicher wieder freigeben
if cv2.waitKey(0) & 0xff == 27: 
    cv2.destroyAllWindows() 