from calibrage.calibrage_multiple_views import getMINT_MEXT1_MEXT2
from utils.take_video import take_video
from utils.take_pictures import take_pictures
import  argparse
import cv2
from tracking.ball_tracking import get_ball_position
from tracking import box_tracking
from reco3D.epip_lines import get_corners
import matplotlib.pyplot as plt
import numpy as np
from reco3D.minimize_dist import get_optimal_points
import imutils
import socket
import json

### Calibrage des caméras
if __name__ == "__main__":

    N_img = 30
    id_cam1 = 6
    id_cam2 = 4
    terrain = False
    send_message = True
    # Serveur
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5065
    # UDP serveur
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # yellowLower = (20, 95, 156)
    # yellowUpper = (90, 235, 255)
    yellowLower = (24, 180, 122)
    yellowUpper = (42, 255, 255)

    args = argparse.ArgumentParser()
    args.add_argument('--show',action='store_true')
    args.add_argument('--save',action='store_true')
    args.add_argument("-v_1", "--video_1", type=str, help="path to input video file", default="your_video_1.avi")
    args.add_argument("-v_2", "--video_2", type=str, help="path to input video file", default="your_video_2.avi")
    args = vars(args.parse_args())
    Mint, dist, Mext1, Mext2, frame_1, frame_2 = getMINT_MEXT1_MEXT2(id_cam1, id_cam2, N_img, args=args)
    print("Calibration done !")
    cv2.destroyAllWindows()

    frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
    objp, corners_cam1 = get_corners(frame_1, (11,8))
    _, corners_cam2 = get_corners(frame_2, (11,8))
    
    # stereo calibration
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objp, corners_cam1, corners_cam2, 
                        Mint, dist, Mint, dist, 
                        (633, 470), flags=cv2.CALIB_FIX_INTRINSIC)
    corners_cam1 = corners_cam1[0].reshape(88,2)
    corners_cam2 = corners_cam2[0].reshape(88,2)

    cap1 = cv2.VideoCapture(id_cam1)
    cap2 = cv2.VideoCapture(id_cam2)
    print("Starting tracking...")
    COINS = []
    COINS2 = []
    COINS3D = []
    PTS_to_SAVE = []
    while(True):
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            np.save("PTS_to_SAVE_LUC.npy", PTS_to_SAVE)
            break
        
        ### add undistort potentiellement

        ball_detected_1, center_1, radius_1 = get_ball_position(frame1, yellowLower, yellowUpper)
        ball_detected_2, center_2, radius_2 = get_ball_position(frame2, yellowLower, yellowUpper)
        
        cpp_is_ball_detected = 0
        X = np.array([0,0,0])
        if ball_detected_1 and ball_detected_2:
            cpp_is_ball_detected = 1

            t_vec_L, t_vec_R = Mext1[:,3], Mext2[:,3]

            x = np.array([center_1[0], center_1[1], 1])
            x_prime = np.array([center_2[0], center_2[1], 1])

            # camera_center
            C1 = np.array([0,0,0])
            C2 = np.array([0,0,0])
            # express C2 in C1 coordinate system
            C2 = -R.transpose() @ T

            C2 = C2.reshape(3,)


            # project points to the camera plane
            x_cam = np.linalg.inv(Mint) @ x # C1 plane
            x_prime_cam = np.linalg.inv(Mint) @ x_prime # C2 plane
            x_prime_cam = np.squeeze(x_prime_cam) 
            # express C2 projected point in C1 plane
            x_prime_cam = R.transpose() @ (x_prime_cam.reshape(3,1) - T)
            x_prime_cam = x_prime_cam.reshape(3,)
            # line from camera center towards the point
            l1 = x_cam
            l2 = x_prime_cam - C2


            # normalize the lines
            l1 = l1 / np.linalg.norm(l1)
            l2 = l2 / np.linalg.norm(l2)

            # get the 3D point with the shortest distance to both lines
            n = np.cross(l1, l2)
            n1 = np.cross(l1, n)
            n2 = np.cross(l2, n)
            # n = n / np.linalg.norm(n)

            c1 = C1 + (np.dot(C2- C1, n2))/np.dot(l1, n2) * l1
            c2 = C2 + (np.dot(C1- C2, n1))/np.dot(l2, n1) * l2

            X = (c1 + c2) / 2

            # print('3D point: ', X)
            
            # plot the lines
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(0, 0, 0, c='b', marker='o')
            # ax.scatter(C2[0], C2[1], C2[2], c='r', marker='o')
            # # ax.scatter(x_prime_cam[0], x_prime_cam[1], x_prime_cam[2], c='g', marker='o')
            # # ax.scatter(c1[0], c1[1], c1[2], c='y', marker='o')
            # # ax.scatter(c2[0], c2[1], c2[2], c='c', marker='o')
            # ax.quiver(0, 0, 0, l1[0], l1[1], l1[2], length=2000, normalize=True, color='b')
            # ax.quiver(C2[0], C2[1], C2[2], l2[0], l2[1], l2[2], length=2000, normalize=True, color='r')
            # ax.scatter(X[0], X[1], X[2], c='c', marker='o')
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # plt.show()
            PTS_to_SAVE.append(X)

            ### test afin de savoir si la balle est dans le terrain
            if terrain:
                if len(COINS) == 0:
                    ret, corners = cv2.findChessboardCorners(frame1, (11,8),None)
                    ret2, corners2 = cv2.findChessboardCorners(frame2, (11,8),None)
                    # cv2.drawChessboardCorners(frame1, (11,8), corners, ret)
                    # cv2.drawChessboardCorners(frame2, (11,8), corners2, ret2)
                
                    if ret and ret2:
                        COINS = [corners[0], corners[10], corners[77], corners[87]]
                        COINS2 = [corners2[0], corners2[10], corners2[77], corners2[87]]
                        cv2.drawChessboardCorners(frame1, (2,2), np.array(COINS), ret)
                        cv2.drawChessboardCorners(frame2, (2,2), np.array(COINS2), ret2)

                        ### get 3D coordinates of the corners
                        for i in range(len(COINS)):
                            x = np.array([COINS[i][0][0], COINS[i][0][1], 1])
                            x_prime = np.array([COINS2[i][0][0], COINS2[i][0][1], 1])
                            x_cam = np.linalg.inv(Mint) @ x
                            x_prime_cam = np.linalg.inv(Mint) @ x_prime
                            x_prime_cam = np.squeeze(x_prime_cam)

                            x_prime_cam = R.transpose() @ (x_prime_cam.reshape(3,1) - T)
                            x_prime_cam = x_prime_cam.reshape(3,)
                            l1 = x_cam 
                            l2 = x_prime_cam - C2 
                            l1 = l1 / np.linalg.norm(l1)
                            l2 = l2 / np.linalg.norm(l2)
                            n = np.cross(l1, l2)
                            n1 = np.cross(l1, n)
                            n2 = np.cross(l2, n)
                            c1 = C1 + (np.dot(C2- C1, n2))/np.dot(l1, n2) * l1
                            c2 = C2 + (np.dot(C1- C2, n1))/np.dot(l2, n1) * l2
                            Xi = (c1 + c2) / 2
                            COINS3D.append(Xi)
                else:
                    ### if X se trouve dans la box de corners COINS3D alors on print ("IN")
                    ### else on print ("OUT")
                        ### on test les coordonnées en x et en z 
                    if X[0] < COINS3D[0][0] and X[0] > COINS3D[1][0] and X[0] < COINS3D[2][0] and X[0] > COINS3D[3][0] and X[2] > COINS3D[0][2] and X[2] > COINS3D[2][2]:
                        print("IN")
                    else:
                        print("OUT")

                    # print("COINS3D : ", COINS3D)
                    
        cv2.imshow('frame1', frame1)
        cv2.imshow('frame2', frame2)
        print("X : ", X)
        av_speed = 0    
        if send_message:
            message = json.dumps({"is_detected": cpp_is_ball_detected, 
                                  "x": float(X[0]),
                                  "y": float(X[1]),
                                  "z": float(X[2]),
                                  "speed": round(av_speed,2),
                                  })
            sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

    cap1.release()
    cap2.release()


