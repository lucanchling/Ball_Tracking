import cv2
import numpy as np
import matplotlib.pyplot as plt

from calib_temp import calibrate_proper

def drawlines(img1, img2, lines, pts1, pts2):
    """draw the epilines"""
    r,c = img1.shape
    for r,pt1,pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1]])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1.astype(int)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2.astype(int)), 5, color, -1)
    return img1,img2

if __name__ == '__main__':
    imname = "mire.png"
    imgL = cv2.imread('Pictures/C1/'+imname,0)
    imgR = cv2.imread('Pictures/C2/'+imname,0)

    objpoints, imgpoints, mtx1, dist1, rvecs1, tvecs1 = calibrate_proper(imgL)
    objpoints, imgpoints2, mtx2, dist2, rvecs2, tvecs2 = calibrate_proper(imgR)

    # stereo calibration
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints, imgpoints2, 
                        mtx1, dist1, mtx2, dist2, 
                        imgL.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC)

    # stereo rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(M1, d1, M2, d2, 
                        imgL.shape[::-1], R, T, alpha=0)

    # getoptimalnewcameramatrix and undistort
    newM1, roi1 = cv2.getOptimalNewCameraMatrix(M1, d1, imgL.shape[::-1], 1, imgL.shape[::-1])
    newM2, roi2 = cv2.getOptimalNewCameraMatrix(M2, d2, imgL.shape[::-1], 1, imgL.shape[::-1])

    new_d1 = cv2.undistort(imgL, M1, d1, None, newM1)
    new_d2 = cv2.undistort(imgR, M2, d2, None, newM2)

    # compute the epilines
    lines1 = cv2.computeCorrespondEpilines(imgpoints[0].reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    lines2 = cv2.computeCorrespondEpilines(imgpoints2[0].reshape(-1,1,2), 2, F)
    lines2 = lines2.reshape(-1,3)

    img5,img6 = drawlines(imgL, imgR, lines1, imgpoints[0].reshape(-1,2), imgpoints2[0].reshape(-1,2))
    img3,img4 = drawlines(imgR, imgL, lines2, imgpoints2[0].reshape(-1,2), imgpoints[0].reshape(-1,2))

    cv2.imshow('L', img3)
    cv2.imshow('R', img4)
    cv2.waitKey(0)