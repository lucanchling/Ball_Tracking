import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import argparse

### Transform a rotation vector and a translation vector into a 4x4 matrix extrinsic
def rotation_vec_to_matrix(r,t):
    Mrot = np.zeros((3, 3))
    Mrot[0,0] = np.cos(r[2])*np.cos(r[1])
    Mrot[0,1] = -np.cos(r[2])*np.sin(r[1])
    Mrot[0,2] = np.sin(r[2])
    Mrot[1,0] = np.sin(r[0])*np.sin(r[2])*np.cos(r[1]) + np.cos(r[0])*np.sin(r[1])
    Mrot[1,1] = -np.sin(r[0])*np.sin(r[2])*np.sin(r[1]) + np.cos(r[0])*np.cos(r[1])
    Mrot[1,2] = -np.sin(r[0])*np.cos(r[2])
    Mrot[2,0] = -np.cos(r[0])*np.sin(r[2])*np.cos(r[1]) + np.sin(r[0])*np.sin(r[1])
    Mrot[2,1] = np.cos(r[0])*np.sin(r[2])*np.sin(r[1]) + np.sin(r[0])*np.cos(r[1])
    Mrot[2,2] = np.cos(r[0])*np.cos(r[2])
    M = np.hstack((Mrot,t))
    M = np.vstack((M,np.array([0,0,0,1])))
    return M


def getMINT_MEXT1_MEXT2(N_img, args):
    id_cam_1 = 4
    id_cam_2 = 6
    Nx = 11
    Ny = 8

    ### Calibration de la camera 1 avec 10 vues différentes
    Timer_tic = time.time()
    cap1 = cv2.VideoCapture(id_cam_1)
    FRAMES = []
    CORNERS = []

    ### Save N_img in order to calibrate the first camera
    while len(FRAMES) < N_img:
        ret1, frame1 = cap1.read()
        ret, corners = cv2.findChessboardCorners(frame1, (Nx,Ny),None)
        cv2.drawChessboardCorners(frame1, (Nx,Ny), corners, ret)
        cv2.imshow('frame1',frame1)
        key = cv2.waitKey(1) & 0xFF
        Timer = 2 * time.time() - Timer_tic - time.time()
        if ret and Timer > 0.5:
            Timer_tic = time.time()
            FRAMES.append(frame1)
            CORNERS.append(corners)
            ### if I tap y, we can continue  if not we are waiting
        if key == ord("q"):
            break

        print(len(FRAMES))
    cv2.destroyAllWindows()

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((Nx*Ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:Nx,0:Ny].T.reshape(-1,2)
    objp = objp * 20
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for i in range(len(CORNERS)):
        objpoints.append(objp)
        gray = cv2.cvtColor(FRAMES[i], cv2.COLOR_BGR2GRAY)
        corners2 = cv2.cornerSubPix(gray,CORNERS[i],(11,11),(-1,-1),criteria)
        imgpoints.append(CORNERS[i])

    ### On force l'apriori sur la matrice intrinsèque
    focal = 4
    i1 = frame1.shape[1]/2
    i2 = frame1.shape[0]/2
    s1 = 0.0049 #0.00489614
    s2 = 0.0049 #s2 = 0.00496559
    f1 = focal/s1
    f2 = focal/s2

    ### Création de la matrice intrinsèque a priori
    M_int_a_priori = np.array([[f1, 0, i1],[0, f2, i2],[0, 0, 1]])

    print("M_int_a_priori :", M_int_a_priori)
    
    ### On obtient Mint qui est la matrice intrinsèque de la caméra 1 et de la caméra 2
    ret, Mint, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], M_int_a_priori, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS+cv2.CALIB_FIX_ASPECT_RATIO+cv2.CALIB_FIX_PRINCIPAL_POINT) #cv2.CALIB_USE_INTRINSIC_GUESS+cv2.CALIB_FIX_FOCAL_LENGTH)
    print("Mint :", Mint)


    ### On prend en compte la distorsion
    undistortion = True
    if undistortion:
        ### Undistort
        h,  w = FRAMES[0].shape[:2]
        newMint, roi = cv2.getOptimalNewCameraMatrix(Mint, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(FRAMES[0], Mint, dist, None, newMint)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        plt.figure()
        plt.imshow(dst)
        plt.title("Undistorted image")
        print("newMint : ", newMint)
    ### On vérifie que le tz correspond bien à la distance entre la caméra 1 et la mire (en mm)
    print("rvecs[0]",rvecs[0])
    print("tvecs[0]",tvecs[0])

    ### On save une frame au même instant t sur les deux caméras de la grille 
    cap2 = cv2.VideoCapture(id_cam_2)
    ret_cam2, frame_cam2 = cap2.read()
    ret_cam1, frame_cam1 = cap1.read()
    ret2, corners_cam2 = cv2.findChessboardCorners(frame_cam2, (Nx,Ny),None)
    ret1, corners_cam1 = cv2.findChessboardCorners(frame_cam1, (Nx,Ny),None)
    while ret_cam2 == False and ret_cam1 == False:
        ret_cam2, frame_cam2 = cap2.read()
        ret_cam1, frame_cam1 = cap1.read()
        ret2, corners_cam2 = cv2.findChessboardCorners(frame_cam2, (Nx,Ny),None)
        ret1, corners_cam1 = cv2.findChessboardCorners(frame_cam1, (Nx,Ny),None)


    plt.figure()
    plt.imshow(frame_cam2)
    plt.title("Image de la mire pour la calibration de la caméra 2")
    plt.figure()
    plt.imshow(frame_cam1)
    plt.title("Image de la mire pour la calibration de la caméra 1")

    ### On obtient les paramètres extrinsèques de la caméra 2 en utilisant MINT et les facteurs de distorsion
    success, rvecs_cam2, tvecs_cam2 = cv2.solvePnP(objp, corners_cam2, newMint, dist)
    proj_cam2 = cv2.projectPoints(objp, rvecs_cam2, tvecs_cam2, newMint, dist)

    plt.figure()
    plt.imshow(frame_cam2)
    plt.scatter(proj_cam2[0][:,:,0], proj_cam2[0][:,:,1])
    plt.title("Projection des points sur l'image de la caméra 2")
    ### création de la matrice extrinsèque de la caméra 2
    Mext_cam2 = rotation_vec_to_matrix(rvecs_cam2, tvecs_cam2)
    print("Mext_cam2 : ", Mext_cam2)

    ### On obtient les paramètres extrinsèques de la caméra 1 en utilisant MINT et les facteurs de distorsion
    success_cam1, rvecs_cam1, tvecs_cam1 = cv2.solvePnP(objp, corners_cam1, newMint, dist)
    proj_cam1 = cv2.projectPoints(objp, rvecs_cam1, tvecs_cam1, newMint, dist)

    plt.figure()
    plt.imshow(frame_cam1)
    plt.scatter(proj_cam1[0][:,:,0], proj_cam1[0][:,:,1])
    plt.title("Projection des points sur l'image de la caméra 1")
    # plt.show()
    ### création de la matrice extrinsèque de la caméra 1
    Mext_cam1 = rotation_vec_to_matrix(rvecs_cam1, tvecs_cam1)
    print("Mext_cam1 : ", Mext_cam1)
    if args['show']:
        print("Showing figures...")
        plt.show()
    if args['save']:
        print("Saving matrices...")

    return newMint, dist, Mext_cam1, Mext_cam2, frame_cam1, frame_cam2


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--show',action='store_true')
    args.add_argument('--save',action='store_true')
    args = vars(args.parse_args())
    Mint, dist, Mext1, Mext2, frame_1, frame_2 = getMINT_MEXT1_MEXT2(N_img=20, args=args)
