import numpy as np

from epip_lines import main as epip_main

def main():
    mtx, Mext_L, Mext_R, R, T, F, ptsL, ptsR, center1, center2, epilineL, epilineR = epip_main(None)
    print('mtx: ', mtx)
    s1 = 0.0049
    f = mtx[0,0] * s1
    print('f: ', f)
    
    B = 320 # Baseline
    camera_center = np.array([mtx[0,2], mtx[1,2]])
    print('camera_center: ', camera_center)
    #### Disparity = distance between the two centers
    center1 = np.asarray(center1)
    center2 = np.asarray(center2)

    center1 = center1 - camera_center
    center2 = center2 - camera_center
    camera_center = camera_center - camera_center
    S = np.array([[1/s1,0],[0,1/s1]])
    print('S: ', S)
    S_inv = np.linalg.inv(S)
    print('S_inv: ', S_inv)

    center1 = S_inv @ center1
    center2 = S_inv @ center2
    camera_center = S_inv @ camera_center



    dist1 = np.sqrt((center1[0]-camera_center[0])**2 + (center1[1]-camera_center[1])**2)
    dist2 = np.sqrt((center2[0]-camera_center[0])**2 + (center2[1]-camera_center[1])**2)
    Disparity = dist1 - dist2
    print('Disparity: ', Disparity)
    Z = f * B / Disparity
    print('Z: ', Z)

    

if __name__ == '__main__':
    main()
