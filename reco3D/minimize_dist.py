import numpy as np
import cv2
from scipy import linalg
import matplotlib.pyplot as plt

from epip_lines import main as epip_main


def get_optimal_points(F, center1, center2):
    centerL = np.reshape(center1, (1,1,2))
    centerR = np.reshape(center2, (1,1,2))

    opt_centerL, opt_centerR = cv2.correctMatches(F, centerL, centerR)

    opt_centerL, opt_centerR = np.squeeze(opt_centerL), np.squeeze(opt_centerR)

    return opt_centerL, opt_centerR

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

def get_3dpts_DLT(mtx, R, T, ptsL, ptsR, center1, center2):
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

def solve_3d_coord(mtx, R, T, ptsL, ptsR):

    ptsL_hmg = np.hstack((ptsL,1)).reshape(3,1)
    ptsR_hmg = np.hstack((ptsR,1)).reshape(3,1)

    # solve the system x = mtx @ X , x_prime = mtx @ (R @ X + T) using least squares
    # to get the 3D point X
    mtx_hmg = np.hstack((mtx, np.zeros((3,1))))

    A = np.vstack((mtx, mtx @ R))
    b = np.vstack((ptsL_hmg, ptsR_hmg - mtx @ T))
    X = np.linalg.lstsq(A, b, rcond=None)[0]

    X = X.reshape(3,1)

    return X

def main():
    mtx, Mext_L, Mext_R, R, T, F, ptsL, ptsR, center1, center2, epilineL, epilineR = epip_main(None)
    
    t_vec_L, t_vec_R = Mext_L[:,3], Mext_R[:,3]

    # get_3dpts_DLT(mtx, R, T, ptsL, ptsR, center1, center2)

    # x = np.array([center1[0], center1[1], 1])
    # x_prime = np.array([center2[0], center2[1], 1])

    x = np.array([ptsL[0][0], ptsL[0][1], 1])
    x_prime = np.array([ptsR[0][0], ptsR[0][1], 1])

    # optimal solution that minimizes the distance between the two lines
    centerL, centerR = get_optimal_points(F, center1, center2)

    center1 = np.asarray(center1)
    center2 = np.asarray(center2)

    # camera_center
    C1 = np.array([0,0,0])
    C2 = -R.transpose() @ T
    C2 = np.squeeze(C2)

    # project points to the camera plane
    x_cam = np.linalg.inv(mtx) @ x # C1 plane
    x_prime_cam = np.linalg.inv(mtx) @ x_prime # C2 plane 


    x_prime_cam_C2 = R.transpose() @ x_prime_cam.reshape(3,1) + T

    # line from camera center towards the point
    l1 = x_cam
    l2 = x_prime_cam - C2


    # normalize the lines
    l1 = l1 / np.linalg.norm(l1)
    l2 = l2 / np.linalg.norm(l2)

    print('l1:',l1)
    print('l2:',l2)

    # get the 3D point with the shortest distance to both lines
    A = np.vstack((l1, l2))
    b = np.array([np.dot(l1,C1), np.dot(l2,C2)])
    X = np.linalg.lstsq(A, b, rcond=None)[0]
    X = X.reshape(3,1)
    

    print('3D point: ', X)


    # plot the lines
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, c='b', marker='o')
    ax.scatter(C2[0], C2[1], C2[2], c='r', marker='o')
    ax.quiver(0, 0, 0, l1[0], l1[1], l1[2], length=20, normalize=True, color='b')
    ax.quiver(C2[0], C2[1], C2[2], l2[0], l2[1], l2[2], length=20, normalize=True, color='r')
    ax.scatter(X[0], X[1], X[2], c='y', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    # X = solve_3d_coord(mtx, R, T, center1, center2)
    # print('3D point: ', X)
    
    
if __name__ == '__main__':
    main()
