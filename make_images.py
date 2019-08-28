import os
import sys

import numpy as np

from get_frames import get_frames

import cv2
import cv2.aruco as aruco



aruco_dict_std = aruco.Dictionary_get(aruco.DICT_4X4_50) # grab a standard dictionary
aruco_dict = aruco.Dictionary_create_from(2,4, aruco_dict_std) # make a dictionary from the standard 4x4, but with only two tags in it
parameters =  aruco.DetectorParameters_create()

def make_images(path):
    frames = get_frames(path, show=False)

    for i in range(1,len(frames)):
        l, r = find_l_r(frames[i])
        l_last, r_last = find_l_r(frames[i-1])
        d = l-r

        frame_last = frames[i-1]

        if(l_last[0] < r_last[0]):
            cv2.arrowedLine(frame_last, tuple(r_last), tuple(l_last-d), (255, 0, 0), 5)
        else:            
            cv2.arrowedLine(frame_last, tuple(l_last), tuple(r_last+d), (255, 0, 0), 5)

        cv2.imshow('Test Frame', cv2.resize(frame_last, (int(1.77*500), int(1.*500))))
        cv2.imwrite(f"out/{i}.jpg", frame_last)
        if cv2.waitKey(0) & 0xFF == ord(' '):
            continue

def corners_to_centers(corners):
    return np.array(corners[0]).mean(axis=1)

def find_l_r(frame):
    [l_corners, r_corners], ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    l = np.array(l_corners[0]).mean(axis=0)
    r = np.array(r_corners[0]).mean(axis=0)
    return l, r

if __name__ == '__main__':
    make_images(sys.argv[1])