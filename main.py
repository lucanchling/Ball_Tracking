from calibrage.calibrage_multiple_views import getMINT_MEXT1_MEXT2
from utils.take_video import take_video
from utils.take_pictures import take_pictures
import  argparse
import cv2
from tracking.ball_tracking import get_ball_position
from tracking import box_tracking


### Calibrage des caméras
if __name__ == "__main__":

    N_img = 15
    id_cam1 = 4
    id_cam2 = 6
    yellowLower = (20, 95, 156)
    yellowUpper = (90, 235, 255)

    args = argparse.ArgumentParser()
    args.add_argument('--show',action='store_true')
    args.add_argument('--save',action='store_true')
    args.add_argument("-v_1", "--video_1", type=str, help="path to input video file", default="your_video_1.avi")
    args.add_argument("-v_2", "--video_2", type=str, help="path to input video file", default="your_video_2.avi")
    args = vars(args.parse_args())
    Mint, dist, Mext1, Mext2 = getMINT_MEXT1_MEXT2(N_img, args=args)
    # take_pictures(id_cam1, id_cam2)
    cv2.destroyAllWindows()

    cap1 = cv2.VideoCapture(id_cam1)
    cap2 = cv2.VideoCapture(id_cam2)
    while(True):
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        cv2.imshow('frame1', frame1)
        cv2.imshow('frame2', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ball_detected_1, center_1, radius_1 = get_ball_position(frame1, yellowLower, yellowUpper)
        ball_detected_2, center_2, radius_2 = get_ball_position(frame2, yellowLower, yellowUpper)
        print("ball_detected_1 : ", ball_detected_1)
        print("ball_detected_2 : ", ball_detected_2)
        print("center_1 : ", center_1)
        print("center_2 : ", center_2)

        




    cap1.release()
    cap2.release()


