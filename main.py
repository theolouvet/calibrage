from PIL import Image
from wxPython._core import wxSIZE_ALLOW_MINUS_ONE

import numpy as np
import matplotlib.pyplot as plt
import cv2




cap = cv2.VideoCapture(0)

ni =0
acq = False
coord_px = []
coord_mm = []
N = 2 * 49

A = np.zeros((N,7))
U = np.zeros((N, 1))
focale = 4.0 #mm
coordim = []

def calibration(i):
    im = []
    print'coucou' + str(i)
    k = 0


    for k in range (0,i ):
        coord_px = []
        coord_mm = []
        print'coucou' + str(k)
        dataim = cv2.imread("image"+str(k)+".jpg")
        retc, corners = cv2.findChessboardCorners(dataim, (7, 7), None)
        dataim = cv2.drawChessboardCorners(dataim, (7, 7), corners, retc)
        for kp in corners:
            coord_px.append([kp[0][0], kp[0][1]])
        print coord_px
        for ko in range(1,8):
            for it in range(1 , 8):
                coord_mm.append([20*it, 20 * ko, k * 40])
        im.append([dataim, coord_px, coord_mm, k * 40])
        cv2.imshow('acq'+str(k),im[k][0])
    i = 0
    j = 0

    l, h, w = dataim.shape
    i_co = [h/2, l/2]
    for j in range(0,2):
        print str(A[0][0])
        ind = (j ) * (N / 2)
        for i in range(0,N/2):
            u = []
            u.append(im[j][1][i][0] - i_co[0])
            u.append(im[j][1][i][1] - i_co[1])
            U[ind][0] =  u[0]
            A[ind][0] = u[1]* im[j][2][i][0]
            A[ind][1] = u[1] * im[j][2][i][1]
            A[ind][2] = u[1] * im[j][2][i][2]
            A[ind][3] = u[1]
            A[ind][4] = -u[0]* im[j][2][i][0]
            A[ind][5] = -u[0] * im[j][2][i][1]
            A[ind][6] = -u[0] * im[j][2][i][2]

            ind = ind + 1





    print(A.shape)
    print(U.shape)
    L = np.dot(np.linalg.pinv(A), U)
    print 'lkuyeguyfegtufeygtufeu'
    print(L)
    print(len(L))
    print(len(L[0]))
    oc2 = 1/(np.sqrt(pow(L[4],2)+pow(L[5],2)+pow(L[6],2)))
    beta = oc2 * np.sqrt(pow(L[0],2) + pow(L[1],2)+ pow(L[2],2))
    oc1 = L[3]*oc2/beta
    r11=L[0]*oc2/beta
    r12=L[1]*oc2/beta
    r13=L[2]*oc2/beta
    r21=L[4]*oc2
    r22=L[5]*oc2
    r23=L[6]*oc2
    r1 = np.array([r11[0] , r12[0] , r13[0]])
    r2=np.array([r21[0] , r22[0] , r23[0]])
    r3 = np.cross(r1,r2)
    r31=r3[0]
    r32=r3[1]
    r33=r3[2]
    w = np.arcsin(r13)
    gamma = np.arccos(r11 / np.cos(w))
    phi = np.arccos(r33 / np.cos(w))
    print(str(w))
    print(str(gamma))
    print(str(phi))





while(True):
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,200,250)
    retc, corners = cv2.findChessboardCorners(gray , (7,7), None)
    cv2.imshow('Capture_Video2 ', frame)

    if (retc == True ):
        img = frame
        #cv2.drawChessboardCorners(img, (7, 7), corners, retc)
        cv2.imshow('img', img)


    key=cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        #cv2.imwrite("image"+str(ni)+".jpg", frame)
        ni = ni + 1
        k = 0
        #coord_px.append(corners)
        for k in corners:
            coord_px.append([k[0][0], k[0][1]])
        print coord_px
        for k in range(1,8):
            for i in range(1 , 8):
                coord_mm.append([20*i, 20 * k])

        print coord_mm
    elif key & 0xFF == ord('a'):
        calibration(2)



cap.release()