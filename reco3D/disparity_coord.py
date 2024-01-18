import numpy as np

from epip_lines import main as epip_main

def main():
    mtx, R, T, F, ptsL, ptsR, center1, center2, epilineL, epilineR = epip_main(None)
    
    s1 = 0.0049
    f = mtx[0,0] * s1
    
    B = 320 # Baseline

if __name__ == '__main__':
    main()
