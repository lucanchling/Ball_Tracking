import numpy as np
import cv2
from matplotlib import pyplot as plt
from plotly import express as px
import time
from calib_fct import rotation_vec_to_matrix, cv2Calibrate



id_cam_1 = 4
id_cam_2 = 6
N_img = 200
Nx = 11
Ny = 8
### Calibration de la camera 1 avec 10 vues différentes
Timer_tic = time.time()
cap1 = cv2.VideoCapture(id_cam_1)
FRAMES = []
CORNERS = []
while len(FRAMES) < N_img:

    ret1, frame1 = cap1.read()
    ret, corners = cv2.findChessboardCorners(frame1, (Nx,Ny),None)
    cv2.drawChessboardCorners(frame1, (Nx,Ny), corners, ret)
    cv2.imshow('frame1',frame1)
    key = cv2.waitKey(1) & 0xFF
    Timer = 2 * time.time() - Timer_tic - time.time()
    if ret and Timer > 0.5:
        Timer_tic = time.time()
        # cv2.drawChessboardCorners(frame1, (Nx,Ny), corners, ret)
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

ret, Mint, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


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
    # print("newMint : ", newMint)



### Calibration de la camera 2 avec 1 vue de la mire pour obtenir Mext_cam2
cap2 = cv2.VideoCapture(id_cam_2)
ret_cam2, frame_cam2 = cap2.read()
ret, corners_cam2 = cv2.findChessboardCorners(frame_cam2, (Nx,Ny),None)
while ret == False:
    ret_cam2, frame_cam2 = cap2.read()
    ret, corners_cam2 = cv2.findChessboardCorners(frame_cam2, (Nx,Ny),None)
# cv2.drawChessboardCorners(frame_cam2, (Nx,Ny), corners_cam2, ret)
plt.figure()
plt.imshow(frame_cam2)


proj_cam2, proj_cam2_old, Mint_cam2, dist_cam2, rvecs_cam2, tvecs_cam2 = cv2Calibrate(corners_cam2, objp, newMint, dist,frame_cam2.shape[1],frame_cam2.shape[0])
plt.figure()
plt.imshow(frame_cam2)
plt.scatter(proj_cam2[0][:,:,0], proj_cam2[0][:,:,1])
plt.scatter(proj_cam2_old[0][:,:,0], proj_cam2_old[0][:,:,1])
MSE = np.mean((proj_cam2[0] - proj_cam2_old[0])**2)
print("MSE :", MSE)

Mext_cam2 = rotation_vec_to_matrix(rvecs_cam2[0], tvecs_cam2[0])
# print("Mint_cam2 :", Mint_cam2)
# print("Mext_cam2 : ", Mext_cam2)


ret_cam1, frame_cam1 = cap1.read()
ret, corners_cam1 = cv2.findChessboardCorners(frame_cam1, (Nx,Ny),None)
while ret == False:
    ret_cam1, frame_cam1 = cap1.read()
    ret, corners_cam1 = cv2.findChessboardCorners(frame_cam1, (Nx,Ny),None)
# cv2.drawChessboardCorners(frame_cam1, (Nx,Ny), corners_cam1, ret)
plt.figure()
plt.imshow(frame_cam1)
proj_cam1, proj_cam1_old, Mint_cam1, dist_cam1, rvecs_cam1, tvecs_cam1 = cv2Calibrate(corners_cam1, objp, newMint, dist,frame_cam1.shape[1],frame_cam1.shape[0])
plt.figure()
plt.imshow(frame_cam1)
plt.scatter(proj_cam1[0][:,:,0], proj_cam1[0][:,:,1])
plt.scatter(proj_cam1_old[0][:,:,0], proj_cam1_old[0][:,:,1])
MSE = np.mean((proj_cam1[0] - proj_cam1_old[0])**2)
print("MSE :", MSE)
Mext_cam1 = rotation_vec_to_matrix(rvecs_cam1[0], tvecs_cam1[0])


plt.show()

