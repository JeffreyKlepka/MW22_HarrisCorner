# from tkinter import image_names
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from matplotlib import image as mpimg

def harris(img_dir,window_size,k,threshold):
    # Schritt 0 - Bild laden und in Graustufen umwandeln
    img = cv2.imread(img_dir)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Gaußfilter um Rauschen zu glätten
    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
    # Dimensionen des Bildes auslesen    
    height = img.shape[0]   #.shape[0] outputs height 
    width = img.shape[1]    #.shape[1] outputs width .shape[2] outputs color channels of image
    matrix_R = np.zeros((height,width))

    # Schritt 1 - partielles Ableiten in x und y Richtung
    dx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=3)
    # dy, dx = np.gradient(gray)

    #   Step 2 - Calculate product and second derivatives (dx2, dy2 e dxy)
    # Schritt 2 - die Koeffizienten der Matrix M bestimmen
    dx2=np.square(dx)
    dy2=np.square(dy)
    dxy=dx*dy

    offset = int( window_size / 2 )
    #   Step 3 - Calcular a soma dos produtos das derivadas para cada pixel (Sx2, Sy2 e Sxy)
    #   Step 4 - Define the matrix H(x,y)=[[Sx2,Sxy],[Sxy,Sy2]]
    #   Step 5 - Calculate the response function ( R=det(H)-k(Trace(H))^2 )
    # Schritt 3 - "das Guckfenster" durch das Bild iterieren
    print ("Ecken finden...")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            Sx2 = np.sum(dx2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sy2 = np.sum(dy2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(dxy[y-offset:y+1+offset, x-offset:x+1+offset])

            # Matrix nach Verschiebung (Shift)
            H = np.array([[Sx2,Sxy],[Sxy,Sy2]])
            
            # Jetzt normalerweise Eigenwerte berechnen 

            # Schritt 5 - Response function berechnen
            det=np.linalg.det(H)  # Determinante der Matrix
            tr=np.matrix.trace(H)  # Spur der Matrix
            R=det-k*(tr**2) # Response Function
            matrix_R[y-offset, x-offset]=R
    
    #   Step 6 - Apply a threshold
    # Einen Grenzwert (threshold) festlegen und danach Punkte selektieren
    cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            value=matrix_R[y, x]
            if value>threshold:
                cv2.circle(img,(x,y),3,(0,255,0))

    '''
    # Bilder darstellen
    img_copy = cv2.imread(img_dir) # Zur Veranschaulichung Ausgangsbild zeigen
    fig = plt.figure(constrained_layout=True)
    ax_array = fig.subplots(2,2, squeeze=False)
    # Darstellung 4 in 1
    ax_array[0,0].imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    ax_array[0,1].imshow(cv2.cvtColor(img_gaussian, cv2.COLOR_BGR2RGB))
    ax_array[1,0].imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    ax_array[1,1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    '''
    # Darstellung Einzelbild
    plt.figure("Harris-Detector - from scratch")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Harris-Detektor - Beispiel 1")
    # plt.savefig('My_harris_detector-thresh_%s.png'%(threshold), bbox_inches='tight')
    plt.show()

harris("testbild1.jpg", 3, 0.04, 0.25) # Change this path to one that will lead to your image