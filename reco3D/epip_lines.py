import cv2
import numpy as np
import argparse
from scipy import linalg

from ball_tracking_temp import get_ball_position

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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

def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    # print(Vh[3,0:3]/Vh[3,3])
    return np.array(Vh[3,0:3]/Vh[3,3])

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

    get_epilines(colorBall_imgL, colorBall_imgR, ptsL, ptsR, F)

    P1 = np.hstack((np.eye(3,3), np.zeros((3,1))))
    P1 = mtx @ P1
    P2 = np.hstack((R, T))
    P2 = mtx @ P2

    pts3d = []

    for i in range(0, len(ptsL)):
        pts3d.append(DLT(P1, P2, ptsL[i], ptsR[i]))

    pts3d = np.asarray(pts3d)
    origin = np.asarray(pts3d[0])
    pts3d = pts3d - origin

    point1 = np.asarray([center1[0], center1[1], 1])
    point2 = np.asarray([center2[0], center2[1], 1])

    ball = np.asarray(DLT(P1, P2, point1, point2))
    ball = ball - origin
    print('origin: ', origin)
    print('ball position: ', ball)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2], c='r', marker='o')
    ax.scatter(ball[0], ball[1], ball[2], c='b', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-u', '--undistort', action='store_true')
    args = vars(args.parse_args())
    
    main(args)