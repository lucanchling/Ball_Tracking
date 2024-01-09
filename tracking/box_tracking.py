# Different Imports
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

def get_tracker(tracker_name="kcf"):
    """Get the tracker from the tracker name"""
    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]
    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
    # function to create our object tracker
    if int(major) == 3 and int(minor) < 3:
        tracker = cv2.Tracker_create(tracker_name.upper())
    # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
    # approrpiate object tracker constructor:
    else:
        # initialize a dictionary that maps strings to their corresponding
        # OpenCV object tracker implementations
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.legacy.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.legacy.TrackerBoosting_create,
            "mil": cv2.legacy.TrackerMIL_create,
            "tld": cv2.legacy.TrackerTLD_create,
            "medianflow": cv2.legacy.TrackerMedianFlow_create,
            "mosse": cv2.legacy.TrackerMOSSE_create
        }
        # grab the appropriate object tracker using our dictionary of
        # OpenCV object tracker objects
        tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
    
    return tracker

def get_video_stream(videoName=None):
    """Get the video stream from the video name"""
    # if a video path was not supplied, grab the reference to the web cam
    if videoName is None:
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
    # otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(videoName)
    
    return vs

def track_bbox(frame, trackerName, tracker, initBB, fps=None):
    """Track the bounding box in the given frame"""
    (H, W) = frame.shape[:2]
        # check to see if we are currently tracking an object
    if initBB is not None:
            # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)
        
        info = [
                ("Tracker", trackerName),
                ("Success", "Yes" if success else "No")
            ]
        
        if fps is not None:
            # update the FPS counter
            fps.update()
            fps.stop()
            try:
                info.append(("FPS", "{:.2f}".format(fps.fps())))
            except ZeroDivisionError:
                pass
            # initialize the set of information we'll be displaying on
            # the frame
            # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        return success

def main(args):
    # get the tracker
    tracker = get_tracker(args["tracker"])

    # initialize the bounding box coordinates of the object we are going
    # to track
    initBB = None

    # get the video stream
    videoName = args["video"]
    vs = get_video_stream(videoName)

    # initialize the FPS throughput estimator
    fps = None

    # loop over frames from the video stream
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = vs.read()
        frame = frame[1] if videoName is not None else frame
        # check to see if we have reached the end of the stream
        if frame is None:
            break
        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        frame = imutils.resize(frame, width=500)
        ret = track_bbox(frame, args["tracker"], tracker, initBB, fps)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                showCrosshair=True)
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()
        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break
    # if we are using a webcam, release the pointer
    if not videoName:
        vs.stop()
    # otherwise, release the file pointer
    else:
        vs.release()
    # close all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":       
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--video", help="path to the video file", default=None)
    args.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
    args = vars(args.parse_args())

    main(args)