import cv2
import argparse

def main(args):

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args["video"], fourcc, 20.0, size)

    while(True):
        _, frame = cap.read()
        cv2.imshow('Recording...', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--video", type=str, help="path to input video file", default="your_video.avi")
    args = vars(args.parse_args())

    main(args)