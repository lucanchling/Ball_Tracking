# import the necessary packages
from collections import deque
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import cv2
import imutils
import time

from box_tracking import get_tracker, track_bbox

def main(args):
    if args["cv2_tracking"]:
        trackerName = args["tracker"]
        tracker = get_tracker(trackerName)
    videoName = args["video"]
    pts = deque(maxlen=64)

    if videoName is None:   # for webcam tennis ball
        yellowLower = (20, 95, 156)
        yellowUpper = (90, 235, 255)
        vs = VideoStream(src=0).start()
    
    else :  # for video tennis ball
        yellowLower = (30, 90, 225)
        yellowUpper = (71, 255, 255)
        vs = cv2.VideoCapture(videoName)
    
    time.sleep(1.0)

    # initialize fps and bounding box
    fps, initBB = None, None

    is_tracking = False

    while True:
        frame = vs.read()
        frame = frame[1] if videoName is not None else frame

        if frame is None:
            break

        frame = imutils.resize(frame, width=500)
        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, yellowLower, yellowUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:   # if there is a ball
            c = max(cnts, key=cv2.contourArea)
            ((x,y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]),
                      int(M["m01"]/M["m00"]))
            
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)),
                           int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

        if args["cv2_tracking"]:
            is_tracking = track_bbox(frame, trackerName, tracker, initBB, fps)
        pts.appendleft(center)

        if args["line"]:
            for i in range(1, len(pts)):
                if pts[i-1] is None or pts[i] is None:
                    continue
                
                thickness = int(np.sqrt(64/float(i+1))*2.5)
                cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and args["cv2_tracking"]:
            tracker = get_tracker(trackerName)
            add_size = 10
            x_min = int(x-radius - add_size) if int(x-radius - add_size) > 0 else 0
            y_min = int(y-radius - add_size) if int(y-radius - add_size) > 0 else 0
            side = int(radius*2 + add_size)
            initBB = (x_min, y_min, side, side)
            print(initBB)
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()

        if key == ord("q"):
            break

    if videoName is None:
        vs.stop()

    else:
        vs.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--video", help="path to the video file", default=None)
    args.add_argument("-l", "--line", help="draw line", action="store_true")
    args.add_argument("-t", "--tracker", type=str, default="csrt",
                      help="OpenCV object tracker type")
    args.add_argument("-ct", "--cv2_tracking", help="use cv2 tracking", action="store_true")
    args = vars(args.parse_args())
    
    main(args)