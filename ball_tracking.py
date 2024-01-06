# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

def main(args):
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

    while True:
        frame = vs.read()
        frame = frame[1] if videoName is not None else frame

        if frame is None:
            break

        frame = imutils.resize(frame, width=600)
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

        pts.appendleft(center)

        if args["line"]:
            for i in range(1, len(pts)):
                if pts[i-1] is None or pts[i] is None:
                    continue
                
                thickness = int(np.sqrt(64/float(i+1))*2.5)
                cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

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
    args = vars(args.parse_args())
    
    main(args)