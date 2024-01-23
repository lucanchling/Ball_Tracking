import cv2
import argparse

def take_video(id_cam_1, id_cam_2,args):

    cap_1 = cv2.VideoCapture(id_cam_1)
    cap_2 = cv2.VideoCapture(id_cam_2)
    width_1 = int(cap_1.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height_1 = int(cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    width_2 = int(cap_2.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height_2 = int(cap_2.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size_1 = (width_1, height_1)
    size_2 = (width_2, height_2)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_1 = cv2.VideoWriter(args["video_1"], fourcc, 20.0, size_1)
    out_2 = cv2.VideoWriter(args["video_2"], fourcc, 20.0, size_2)


    while(True):
        _, frame_1 = cap_1.read()
        _, frame_2 = cap_2.read()
        cv2.imshow('Recording _frame_1...', frame_1)
        cv2.imshow('Recording _frame_2...', frame_2)
        out_1.write(frame_1)
        out_2.write(frame_2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_1.release()
    out_1.release()
    cap_2.release()
    out_2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-v_1", "--video_1", type=str, help="path to input video file", default="your_video_1.avi")
    args.add_argument("-v_2", "--video_2", type=str, help="path to input video file", default="your_video_2.avi")
    args = vars(args.parse_args())
    id_cam_1 = 4
    id_cam_2 = 6
    take_video(id_cam_1, id_cam_2,args)