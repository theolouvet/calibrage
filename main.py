from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2




cap = cv2.VideoCapture(0)

i =0
acq = False
coord_px = []
coord_mm = []
A = np.zeros((7,7))
focale = 4.4 #mm

while(True):
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,200,250)
    retc, corners = cv2.findChessboardCorners(gray , (7,7), None)
    cv2.imshow('Capture_Video2 ', frame)

    if (retc == True ):
        img = frame
        cv2.drawChessboardCorners(img, (7, 7), corners, retc)
        cv2.imshow('img', img)





    key=cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        i = i + 1
        cv2.imwrite("image"+str(i)+".jpg", img)
        k = 0
        #coord_px.append(corners)
        for k in corners:
            coord_px.append([k[0][0], k[0][1]])
        print coord_px
        for k in range(1,8):
            for i in range(1 , 8):
                coord_mm.append([2*i, 2 * k])

        print coord_mm

        for k in range(1,8):
            



cap.release()