import cv2
import numpy as np
import argparse
import imutils

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def get_ball_position(frame, lower, upper, draw=True):
    is_detected = False
    center = None
    radius = None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    cnts = cv2.findContours(mask.copy(),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:   # if there is a ball
        c = max(cnts, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]),
                    int(M["m01"]/M["m00"]))
        
        if radius > 10:
            if draw:
                cv2.circle(frame, (int(x), int(y)),
                            int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

            is_detected = True

    return is_detected, center, radius

def drawPoints(img, pts, colors):
    for pt, color in zip(pts, colors):
        cv2.circle(img, tuple(pt.astype(int)), 5, tuple(color.tolist()), -1)

def drawLines(img, lines, colors):
    _, c, _ = img.shape
    for r, color in zip(lines, colors):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img, (x0, y0), (x1, y1), tuple(color.tolist()), 1)

def load_images(path):
    gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    color_img = cv2.imread(path)

    return color_img, gray_img

def undistort(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    x,y,w,h = roi
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    dst = dst[y:y+h, x:x+w]
    return dst

def load_undistorted_images(path, mtx, dist):
    color_img, img = load_images(path)
    img = undistort(img, mtx, dist)
    color_img = undistort(color_img, mtx, dist)

    return color_img, img

def get_corners(img, chessboardsize):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
   
    objp = np.zeros((chessboardsize[0]*chessboardsize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardsize[0],0:chessboardsize[1]].T.reshape(-1,2) * 20

    objpoints = []
    imgpoints = []

    ret, corners = cv2.findChessboardCorners(img, chessboardsize,None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(np.squeeze(corners2,axis=1))

    return objpoints, imgpoints

def get_epilines(imgL, imgR, ptsL, ptsR, F , show=False):
    colors = np.random.randint(0,255,(100,3))

    nb_pts = ptsL.shape[0]

    drawPoints(imgL, ptsL, colors[nb_pts:nb_pts*2])
    drawPoints(imgR, ptsR, colors[0:nb_pts])

    # compute the epilines
    epilinesL = cv2.computeCorrespondEpilines(ptsR.reshape(-1,1,2), 2, F).reshape(-1,3)
    drawLines(imgL, epilinesL, colors[0:3])

    epilinesR = cv2.computeCorrespondEpilines(ptsL.reshape(-1,1,2), 1, F).reshape(-1,3)
    drawLines(imgR, epilinesR, colors[3:6])

    if show:
        cv2.imshow("left", imgL)
        cv2.imshow("right", imgR)
        cv2.waitKey(0)

    return epilinesL, epilinesR
    
def main(args):

    mtx = np.load('data/Mint.npy')
    dist = np.load('data/dist.npy')
    Mext_L = np.load('data/Mext_1.npy')
    Mext_R = np.load('data/Mext_2.npy')
    
    # print('Mext_L: ', Mext_L)
    # print('Mext_R: ', Mext_R)

    color_imgL, imgL = load_undistorted_images('Pictures/C1/grid.png', mtx, dist)
    color_imgR, imgR = load_undistorted_images('Pictures/C2/grid.png', mtx, dist)

    # get the corners
    objpoints, imgpoints = get_corners(imgL, (11,8))
    _, imgpoints2 = get_corners(imgR, (11,8))

    # stereo calibration
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints, imgpoints2, 
                        mtx, dist, mtx, dist, 
                        imgL.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC)
    
    # select some points
    ptsL = np.asarray([imgpoints[0][0], imgpoints[0][1], imgpoints[0][2], imgpoints[0][3], imgpoints[0][4], 
                       imgpoints[0][11], imgpoints[0][12], imgpoints[0][13], imgpoints[0][14], imgpoints[0][15],
                       imgpoints[0][22], imgpoints[0][23], imgpoints[0][24], imgpoints[0][25], imgpoints[0][26]])
    ptsR = np.asarray([imgpoints2[0][0], imgpoints2[0][1], imgpoints2[0][2], imgpoints2[0][3], imgpoints2[0][4],
                       imgpoints2[0][11], imgpoints2[0][12], imgpoints2[0][13], imgpoints2[0][14], imgpoints2[0][15],
                       imgpoints2[0][22], imgpoints2[0][23], imgpoints2[0][24], imgpoints2[0][25], imgpoints2[0][26]])

    get_epilines(color_imgL, color_imgR, ptsL, ptsR, F)

    colorBall_imgL, BallimgL = load_undistorted_images('Pictures/C1/with_measures.png', mtx, dist)
    colorBall_imgR, BallimgR = load_undistorted_images('Pictures/C2/with_measures.png', mtx, dist)

    yellowLower = (26, 120, 101)
    yellowUpper = (43, 255, 255)

    ret, center1, radius1 = get_ball_position(colorBall_imgL, yellowLower, yellowUpper)
    ret, center2, radius2 = get_ball_position(colorBall_imgR, yellowLower, yellowUpper)

    # ptsL = np.asarray([center1])
    # ptsR = np.asarray([center2])

    epilineL, epilineR = get_epilines(colorBall_imgL, colorBall_imgR, ptsL, ptsR, F)

    return mtx, Mext_L, Mext_R, R, T, F, ptsL, ptsR, center1, center2, epilineL, epilineR


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-u', '--undistort', action='store_true')
    args = vars(args.parse_args())
    
    main(args)