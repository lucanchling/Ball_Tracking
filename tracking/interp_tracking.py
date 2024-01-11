import cv2
import argparse
import time
import os
import numpy as np

from ball_tracking import get_ball_position

def get_positions(videoName, yellowLower, yellowUpper):
    cap = cv2.VideoCapture(videoName)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"fps: {fps}")
    print(f"nb_frames: {nb_frames}")
    
    posFileName = videoName.split(".")[0] + ".txt"

    if not os.path.exists(posFileName):
        positions = []
        for i_frame in range(nb_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
            _, frame = cap.read()
            ret , center, radius = get_ball_position(frame, yellowLower, yellowUpper)
            positions.append(center)

        with open(posFileName, "w") as f:
            f.write(f"{positions}\n")
    else:
        with open(posFileName, "r") as f:
            positions = f.readlines()
            positions = [eval(pos) for pos in positions][0]
    return positions

def main(args):
    # HSV color space
    yellowLower = (20, 95, 156)
    yellowUpper = (90, 235, 255)
    
    videoName = args["video"]    
    cap = cv2.VideoCapture(videoName)

    positions = get_positions(videoName, yellowLower, yellowUpper)

    N = 20 # number of frame (before and after) to interpolate

    for i_frame in range(N, len(positions)-N):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        _, frame = cap.read()
        center = positions[i_frame]

        if center is None:
            # print(f"Ball not detected at frame {i_frame}")
            # interpolate the missing center
            before = [center for center in positions[i_frame-N:i_frame] if center is not None]
            after = [center for center in positions[i_frame+1:i_frame+N+1] if center is not None]
            if len(before) == 0 or len(after) == 0:
                continue
            center = (int(np.mean([c[0] for c in before + after])), int(np.mean([c[1] for c in before + after])))
            
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # wait the time between two frames
        time.sleep(1/cap.get(cv2.CAP_PROP_FPS))

        if key == ord("q"):
            break

    cap.release()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--video", type=str,
        help="path to input video file", default="Video/test2.avi")
    args = vars(args.parse_args())

    main(args)