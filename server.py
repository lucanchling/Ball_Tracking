import socket
import cv2
import imutils
import json
import time
import argparse
import numpy as np

from collections import deque

from tracking.ball_tracking import get_ball_position
    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--video", help="path to the video file", default=None)
    args.add_argument("-l", "--line", help="draw line", action="store_true")
    args.add_argument("-t", "--tracker", type=str, default="csrt",
                      help="OpenCV object tracker type")
    args.add_argument("-ct", "--cv2_tracking", help="use cv2 tracking", action="store_true")
    args = vars(args.parse_args())
    
    # Serveur
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5065

    s1 = 0.0049
    yellowLower = (20, 95, 156)
    yellowUpper = (90, 235, 255)
    vs = imutils.video.VideoStream(src=0).start()
    
    # UDP serveur
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    pts = deque(maxlen=32)
    timer = time.time()

    while True:
        frame = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=400)
        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        ball_detected, center, radius = get_ball_position(frame, yellowLower, yellowUpper)
        av_speed = 0
        if center is not None: #and (time.time() - timer > 1):
            timer = time.time()
            x, y = list(center)
            new_center = np.array([np.array([x,y]), time.time()],dtype=object)
            pts.append(new_center)

            if len(pts) > 1:
                for i in range(1,len(pts)):
                    dist = np.linalg.norm(pts[i][0] - pts[0][0])
                    dt = pts[i][1] - pts[0][1]
                    speed = dist / dt
                    # print(speed)
                    av_speed += speed
                av_speed /= len(pts)
        else: 
            x,y = (None, None)

        cpp_is_ball_detected = 1 if ball_detected else 0


            
        message = json.dumps({"is_detected": cpp_is_ball_detected, 
                              "x": x,
                              "y": y,
                              "z": radius,
                              "speed": round(av_speed * s1,2),
                              })

        sock.sendto(message.encode(), (UDP_IP, UDP_PORT))       

        # cv2.imshow("frame", frame)
        # key = cv2.waitKey(1) & 0xFF

        # if key == ord("q"):
        #     break


    vs.stop()

    cv2.destroyAllWindows()




