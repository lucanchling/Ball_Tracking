import numpy as np
import cv2
from matplotlib import pyplot as plt
from plotly import express as px


def calib (frame):
    ret1, corners1 = cv2.findChessboardCorners(frame,(11,8))
    if not ret1:
        print("Erreur - Pas de grille trouvée")
        return
    corners1_squeeze = np.squeeze(corners1, axis=1)
    coord_px = corners1_squeeze

    coord_mm = []
    Nx,Ny = 11,8
    x,y=0,0
    for z in [0]:    
        for y in range(Ny):
            for x in range(Nx):
                coord_mm.append(np.array([x*-20,y*-20,z]).astype(np.float32))
    coord_mm = np.array(coord_mm)

    #centre optique se situant au centre de l'image 1
    i1 = frame.shape[1]/2
    i2 = frame.shape[0]/2

    #création de la matrice A de taille (Nx7) et le vecteur unitaire U1 de taille (Nx1)
    Nx = 8
    Ny = 11
    A = np.zeros((Nx*Ny,7)) # Nx*Ny*2
    U1 = np.zeros((Nx*Ny,1))
    coord_px_center = coord_px-np.array([i1,i2])
    U1=coord_px_center[:,0]
    for i in range(coord_mm.shape[0]):
        A[i,0] = np.multiply(coord_px_center[i,1],coord_mm[i,0])
        A[i,1] = np.multiply(coord_px_center[i,1],coord_mm[i,1])
        A[i,2] = np.multiply(coord_px_center[i,1],coord_mm[i,2])
        A[i,3] = coord_px_center[i,1]
        A[i,4] = np.multiply(-coord_px_center[i,0],coord_mm[i,0])
        A[i,5] = np.multiply(-coord_px_center[i,0],coord_mm[i,1])
        A[i,6] = np.multiply(-coord_px_center[i,0],coord_mm[i,2])

    #Résolution du système linéaire A*L=U1
    L = np.linalg.pinv(A)@U1

    o2c = 1/np.sqrt(L[4]**2+L[5]**2+L[6]**2)
    beta = o2c*np.sqrt(L[0]**2+L[1]**2+L[2]**2)
    o1c = (L[3]*o2c)/beta
    r11 = L[0]*o2c/beta
    r12 = L[1]*o2c/beta
    r13 = L[2]*o2c/beta
    r21 = L[4]*o2c
    r22 = L[5]*o2c
    r23 = L[6]*o2c

    vectoriel = np.cross([r11,r12,r13],[r21,r22,r23])

    r31 = vectoriel[0]
    r32 = vectoriel[1]
    r33 = vectoriel[2]

    phi = -np.arctan(r23/r33)
    gamma = -np.arctan(r12/r11)
    omega = np.arctan(r13/(-r23*np.sin(phi)+r33*np.cos(phi)))


    B = np.zeros((Nx*Ny*2,2))
    R = np.zeros((Nx*Ny*2,1))
    for i in range(coord_mm.shape[0]):
        B[i,0] = coord_px_center[i,1]
        B[i,1] = -(r21*coord_mm[i,0]+r22*coord_mm[i,1]+r23*coord_mm[i,2]+o2c)
        R[i,0] = -coord_px_center[i,1]*(r31*coord_mm[i,0]+r32*coord_mm[i,1]+r33*coord_mm[i,2])

    #Résolution de B(o3c,f2) = R
    o3c,f2 = np.linalg.pinv(B)@R

    f = 4
    s2 = f/f2
    f1 = beta*f2
    s1 = s2/beta


    #taille capteur webcam en mm
    w = s1*frame.shape[0]
    h = s2*frame.shape[1]

    f1=f/s1[0]
    f2 = f/s2[0]

    Mint = np.array([[f1,0,i1,0],[0,f2,i2,0],[0,0,0,1]])
    Mext = np.array([[r11,r12,r13,o1c],[r21,r22,r23,o2c],[r31,r32,r33,o3c[0]],[0,0,0,1]])
    M= Mint@Mext
    alpha = r31*coord_mm[:,0]+r32*coord_mm[:,1]+r33*coord_mm[:,2]+o3c
    quedesuns = np.ones((coord_mm.shape[0],1)) #vecteur de 1 pour ajouter une colonne de 1 à coord_mm pour
                                                # avoir des coordonnées homogènes
    coord_mm_4d = np.concatenate((coord_mm,quedesuns),axis=1)
    aU = []
    for i in range(coord_mm_4d.shape[0]):
        aUi = M@coord_mm_4d[i,:]
        Ui = aUi/alpha[i]
        aU.append(Ui)
    aU = np.array(aU)

    ### MSE
    MSE = np.sqrt(np.sum((aU[:88,:2]-coord_px)**2)/coord_px.shape[0])


    return Mint, Mext


def calibrate_proper(frame):
    M_int, M_ext = calib(frame)


    h,w = frame.shape[:2]
    Nx,Ny = 11,8
    ret, corners = cv2.findChessboardCorners(frame, (Nx,Ny),None)
    if ret == False:
        print("Erreur - Pas de grille trouvée")
        return
    
    corners_squeeze = np.squeeze(corners, axis=1)
    coord_px = corners_squeeze
    imgpoints = [coord_px] 

    objp = np.zeros((Nx*Ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:Nx,0:Ny].T.reshape(-1,2)
    objp = objp * 20
    objpoints = [objp]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h,w) ,M_int,None,None, flags=None )
    
    return mtx, dist, rvecs, tvecs