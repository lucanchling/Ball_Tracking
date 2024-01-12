import cv2
from imutils.video import VideoStream
from matplotlib import pyplot as plt

id_camera = 4
id_cam2 = 6
def take_pictures():
    cap = cv2.VideoCapture(id_camera)
    cap2 = cv2.VideoCapture(id_cam2)
    # Check if the webcam is opened correctly
    ret,frame = cap.read()
    ret2,frame2 = cap2.read()
    while(True):
        ret1, corners1 = cv2.findChessboardCorners(frame,(11,8))
        ret2, corners2 = cv2.findChessboardCorners(frame2,(11,8))
        # cv2.drawChessboardCorners(frame,(11,8),corners1,ret1)
        # cv2.drawChessboardCorners(frame2,(11,8),corners2,ret2)
        cv2.imshow('img1',frame) #display the captured image
        cv2.imshow('img2',frame2)
        if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
            cv2.imwrite('first_cam/c14.png',frame)
            cv2.imwrite('second_cam/c14.png',frame2)
            
            cv2.destroyAllWindows()
            break

    cap.release()

if __name__ == "__main__":
    take_pictures()

    
    
    ###plot corners
    # plt.figure()
    # plt.imshow(frame)
    # plt.scatter(corners1[:,0,0],corners1[:,0,1])
    # plt.figure()
    # plt.imshow(frame2)
    # plt.scatter(corners2[:,0,0],corners2[:,0,1])
    # plt.show()


    # vs1 = VideoStream(src=id_camera).start()
    # vs2 = VideoStream(src=id_cam2).start()
    # while(True):
    #     frame = vs1.read()
    #     frame2 = vs2.read()
    #     # plt.imshow(frame)
    #     # plt.show()
    #     cv2.imshow('img1',frame)
    #     cv2.imshow('img2',frame2)
    # take_pictures()
    