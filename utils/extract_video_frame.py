import cv2 
import argparse
import os
# Function to extract frames 
def FrameCapture(args): 
	if not os.path.exists(args["output"]):
		os.makedirs(args["output"])

	# Path to video file 
	vidObj = cv2.VideoCapture(args["video"]) 

	# Used as counter variable 
	count = 0

	# checks whether frames were extracted 
	success = 1

	while success: 

		# vidObj object calls read 
		# function extract frames 
		success, image = vidObj.read() 

		# Saves the frames with frame-count 
		cv2.imwrite(os.path.join(args["output"],f"frame{count}.jpg"), image) 

		count += 1


# Driver Code 
if __name__ == '__main__': 
	args = argparse.ArgumentParser()
	args.add_argument("-v", "--video", help="path to the video file", default=None)
	args.add_argument("-o", "--output", help="path to the output folder", default=None)
	args = vars(args.parse_args())
	# Calling the function 
	FrameCapture(args) 
