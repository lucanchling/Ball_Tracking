import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

from calib_temp import calibrate_proper
from ball_tracking_temp import get_ball_position

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

def main(args):
    imname = "mire.png"

    color_imgL, imgL = load_images('Pictures/C1/'+imname)
    color_imgR, imgR = load_images('Pictures/C2/'+imname)

    mtx1, dist1, rvecs1, tvecs1 = calibrate_proper(imgL)
    mtx2, dist2, rvecs2, tvecs2 = calibrate_proper(imgR)

    if args['undistort']:
        # undistort the images
        imgL = undistort(imgL, mtx1, dist1)
        imgR = undistort(imgR, mtx2, dist2)
        
        color_imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
        color_imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

    # get the corners
    objpoints, imgpoints = get_corners(imgL, (11,8))
    _, imgpoints2 = get_corners(imgR, (11,8))

    # stereo calibration
    mtx2 = mtx1
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints, imgpoints2, 
                        mtx1, dist1, mtx2, dist2, 
                        imgL.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC)
    
    # select some points
    ptsL = np.asarray([imgpoints[0][0], imgpoints[0][8], imgpoints[0][36]])
    ptsR = np.asarray([imgpoints2[0][5], imgpoints2[0][15], imgpoints2[0][25]])

    colors = np.random.randint(0,255,(100,3))

    drawPoints(color_imgL, ptsL, colors[3:6])
    drawPoints(color_imgR, ptsR, colors[0:3])

    # compute the epilines
    epilinesL = cv2.computeCorrespondEpilines(ptsR.reshape(-1,1,2), 2, F).reshape(-1,3)
    drawLines(color_imgL, epilinesL, colors[0:3])

    epilinesR = cv2.computeCorrespondEpilines(ptsL.reshape(-1,1,2), 1, F).reshape(-1,3)
    drawLines(color_imgR, epilinesR, colors[3:6])


    # cv2.imshow("left", color_imgL)
    # cv2.imshow("right", color_imgR)
    # cv2.waitKey(0)

    num = 3
    colorBall_imgL, BallimgL = load_images('Pictures/C1/im'+str(num)+'.png')
    colorBall_imgR, BallimgR = load_images('Pictures/C2/im'+str(num)+'.png')

    BallimgL = undistort(BallimgL, mtx1, dist1)
    BallimgR = undistort(BallimgR, mtx2, dist2)

    colorBall_imgR = undistort(colorBall_imgR, mtx2, dist2)
    colorBall_imgL = undistort(colorBall_imgL, mtx1, dist1)

    yellowLower = (26, 100, 101)
    yellowUpper = (43, 255, 255)

    ret, center1, radius1 = get_ball_position(colorBall_imgL, yellowLower, yellowUpper)
    ret, center2, radius2 = get_ball_position(colorBall_imgR, yellowLower, yellowUpper)

    ptsL = np.asarray([center1])
    ptsR = np.asarray([center2])

    drawPoints(colorBall_imgL, ptsL, colors[3:6])
    drawPoints(colorBall_imgR, ptsR, colors[0:3])

    epilinesL = cv2.computeCorrespondEpilines(ptsR.reshape(-1,1,2), 2, F).reshape(-1,3)
    drawLines(colorBall_imgL, epilinesL, colors[0:3])

    epilinesR = cv2.computeCorrespondEpilines(ptsL.reshape(-1,1,2), 1, F).reshape(-1,3)
    drawLines(colorBall_imgR, epilinesR, colors[3:6])

    cv2.imshow("left", colorBall_imgL)
    cv2.imshow("right", colorBall_imgR)
    cv2.waitKey(0)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-u', '--undistort', action='store_true')
    args = vars(args.parse_args())
    
    main(args)