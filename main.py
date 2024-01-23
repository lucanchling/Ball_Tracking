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

### Calibrage des caméras
if __name__ == "__main__":

    N_img = 50
    id_cam1 = 4
    id_cam2 = 6
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
    Mint, dist, Mext1, Mext2, frame_1, frame_2 = getMINT_MEXT1_MEXT2(N_img, args=args)
    # take_pictures(id_cam1, id_cam2)
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
    while(True):
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        ### add undistort potentiellement

        frame1 = imutils.resize(frame1, width=400)
        frame2 = imutils.resize(frame2, width=400)
        ball_detected_1, center_1, radius_1 = get_ball_position(frame1, yellowLower, yellowUpper)
        ball_detected_2, center_2, radius_2 = get_ball_position(frame2, yellowLower, yellowUpper)
        
        print("ball_detected_1 : ", ball_detected_1)
        print("ball_detected_2 : ", ball_detected_2)
        if ball_detected_1 and ball_detected_2:
            

            print("center1 : ", center_1)
            print("center2 : ", center_2)
            # cv2.circle(frame1, center_1, int(radius_1), (0, 255, 0), 2)
            # cv2.circle(frame2, center_2, int(radius_2), (0, 255, 0), 2)
            


            t_vec_L, t_vec_R = Mext1[:,3], Mext2[:,3]

            # x = np.array([corners_cam1[1][0], corners_cam1[1][1], 1])
            x = np.array([center_1[0], center_1[1], 1])
            # x = np.array([corners_cam1[]])
            x_prime = np.array([center_2[0], center_2[1], 1])

            # x_prime = np.array([corners_cam2[1][0], corners_cam2[1][1], 1])
            # centerL, centerR = get_optimal_points(F, center_1, center_2)
            # center1 = np.asarray(center_1)
            # center2 = np.asarray(center_2)

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

            print('3D point: ', X)
            
            # plot the lines
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(0, 0, 0, c='b', marker='o')
            ax.scatter(C2[0], C2[1], C2[2], c='r', marker='o')
            # ax.scatter(x_prime_cam[0], x_prime_cam[1], x_prime_cam[2], c='g', marker='o')
            # ax.scatter(c1[0], c1[1], c1[2], c='y', marker='o')
            # ax.scatter(c2[0], c2[1], c2[2], c='c', marker='o')
            ax.quiver(0, 0, 0, l1[0], l1[1], l1[2], length=2000, normalize=True, color='b')
            ax.quiver(C2[0], C2[1], C2[2], l2[0], l2[1], l2[2], length=2000, normalize=True, color='r')
            ax.scatter(X[0], X[1], X[2], c='c', marker='o')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            # plt.show()
        cv2.imshow('frame1', frame1)
        cv2.imshow('frame2', frame2)


    cap1.release()
    cap2.release()


