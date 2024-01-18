import numpy as np
import cv2

from epip_lines import main as epip_main

def main():
    mtx, R, T, F, ptsL, ptsR, center1, center2, epilineL, epilineR = epip_main(None)
    
    # solve the system x = mtx @ X , x_prime = mtx @ (R @ X + T)

    x = np.array([center1[0], center1[1], 1])
    x_prime = np.array([center2[0], center2[1], 1])

    # optimal solution that minimizes the distance between the two lines
    centerL = np.reshape(center1, (1,1,2))
    centerR = np.reshape(center2, (1,1,2))

    opt_centerL, opt_centerR = cv2.correctMatches(F, centerL, centerR)

    






    
if __name__ == '__main__':
    main()
