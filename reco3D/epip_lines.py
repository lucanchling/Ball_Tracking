import cv2
import numpy as np
import matplotlib.pyplot as plt

from calib_temp import calibrate_proper

# draw the provided points on the image
def drawPoints(img, pts, colors):
    for pt, color in zip(pts, colors):
        cv2.circle(img, tuple(pt.astype(int)), 5, tuple(color.tolist()), -1)

# draw the provided lines on the image
def drawLines(img, lines, colors):
    _, c, _ = img.shape
    for r, color in zip(lines, colors):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img, (x0, y0), (x1, y1), tuple(color.tolist()), 1)

if __name__ == '__main__':
    imname = "mire.png"
    color_imgL = cv2.imread('Pictures/C1/'+imname)
    color_imgR = cv2.imread('Pictures/C2/'+imname)

    imgL = cv2.imread('Pictures/C1/'+imname,0)
    imgR = cv2.imread('Pictures/C2/'+imname,0)

    colors = np.random.randint(0,255,(100,3))

    objpoints, imgpoints, mtx1, dist1, rvecs1, tvecs1 = calibrate_proper(imgL)
    objpoints, imgpoints2, mtx2, dist2, rvecs2, tvecs2 = calibrate_proper(imgR)


    # stereo calibration
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints, imgpoints2, 
                        mtx1, dist1, mtx2, dist2, 
                        imgL.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC)

    
    # getoptimalnewcameramatrix and undistort
    newM1, roi1 = cv2.getOptimalNewCameraMatrix(M1, d1, imgL.shape[::-1], 1, imgL.shape[::-1])
    newM2, roi2 = cv2.getOptimalNewCameraMatrix(M2, d2, imgL.shape[::-1], 1, imgL.shape[::-1])

    newdL = cv2.undistort(imgL, M1, d1, None, newM1)
    newdR = cv2.undistort(imgR, M2, d2, None, newM2)
    
    # select some points
    ptsL = np.asarray([imgpoints[0][0], imgpoints[0][25], imgpoints[0][36]])
    ptsR = np.asarray([imgpoints2[0][5], imgpoints2[0][15], imgpoints2[0][25]])

    drawPoints(color_imgL, ptsL, colors[3:6])
    drawPoints(color_imgR, ptsR, colors[0:3])

    # compute the epilines
    epilinesL = cv2.computeCorrespondEpilines(ptsR.reshape(-1,1,2), 2, F).reshape(-1,3)
    drawLines(color_imgL, epilinesL, colors[0:3])

    epilinesR = cv2.computeCorrespondEpilines(ptsL.reshape(-1,1,2), 1, F).reshape(-1,3)
    drawLines(color_imgR, epilinesR, colors[3:6])


    cv2.imshow("left", color_imgL)
    cv2.imshow("right", color_imgR)
    cv2.waitKey(0)