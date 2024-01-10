# import the necessary packages
from collections import deque
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import cv2
import imutils
import time

from box_tracking import get_tracker

def track_bbox(frame, tracker):
    """Track the bounding box in the given frame"""
    (H, W) = frame.shape[:2]
        # check to see if we are currently tracking an object
    if tracker['initBB'] is not None:
            # grab the new bounding box coordinates of the object
        (success, box) = tracker["tracker"].update(frame)
            # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)
        
        info = [
                ("Tracker", tracker["name"]),
                ("Success", "Yes" if success else "No")
            ]
        
        if tracker['fps'] is not None:
            # update the FPS counter
            tracker['fps'].update()
            tracker['fps'].stop()
            try:
                info.append(("FPS", "{:.2f}".format(tracker['fps'].fps())))
            except ZeroDivisionError:
                pass
            # initialize the set of information we'll be displaying on
            # the frame
            # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        tracker['currentBB'] = box
        tracker['is_tracking'] = success

def get_bbox(x,y,radius,add_size=10):
    """Get the bounding box of the ball according to the center and radius"""
    x_min = int(x-radius - add_size) if int(x-radius - add_size) > 0 else 0
    y_min = int(y-radius - add_size) if int(y-radius - add_size) > 0 else 0
    side = int(radius*2 + 2*add_size)
    initBB = (x_min, y_min, side, side)
    # print(initBB)
    return initBB

def start_new_tracking(tracker, frame, x, y, radius):
    tracker["tracker"] = get_tracker(tracker["name"])
    initBB = get_bbox(x,y,radius,15)
    tracker["initBB"] = initBB
    tracker["tracker"].init(frame, initBB)
    tracker['fps'] = FPS().start()

def get_ball_position(frame, lower, upper):
    is_detected = False
    center = None
    radius = None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

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

            is_detected = True

    return is_detected, center, radius

def main(args):
    if args["cv2_tracking"]:
        tracking = {
            'name' : args["tracker"],
            'tracker' : get_tracker(args["tracker"]),
            'initBB' : None,
            'currentBB' : None,
            'fps' : None,
            'is_tracking' : False
        }
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

        frame = imutils.resize(frame, width=400)
        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        ball_detected, center, radius = get_ball_position(frame, yellowLower, yellowUpper)
        x, y = center if center is not None else (None, None)
        
        if ball_detected and args["cv2_tracking"]:
            if not tracking['is_tracking']:
                start_new_tracking(tracking, frame, x, y, radius)
            else:
                centerBB = (tracking['currentBB'][0] + tracking['currentBB'][2]//2, 
                            tracking['currentBB'][1] + tracking['currentBB'][3]//2)
                dist_centers = np.linalg.norm(np.array(center) - np.array(centerBB))
                # print(dist_centers)
                if dist_centers > 50:
                    start_new_tracking(tracking, frame, x, y, radius)

        if args["cv2_tracking"]:
            track_bbox(frame, tracking)
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
            start_new_tracking(tracking, frame, x, y, radius)

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