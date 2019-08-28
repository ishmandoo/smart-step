from __future__ import print_function
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def process_video(path, return_frames_l_i = [], return_frames_r_i = [], show=False, plot=False):
    print(f"processing video {path}")
    
    capture = cv2.VideoCapture(path)
    l = []
    r = []
    both = []

    frame_i = 0

    return_frames_l = []
    return_frames_r = []

    if capture.isOpened():
        frame_captured, frame = capture.read()
    else:
        frame_captured = False

    while frame_captured:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict_std = aruco.Dictionary_get(aruco.DICT_4X4_50) # grab a standard dictionary
        aruco_dict = aruco.Dictionary_create_from(2,4, aruco_dict_std) # make a dictionary from the standard 4x4, but with only two tags in it
        parameters =  aruco.DetectorParameters_create()
        corners_list, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if frame_i in return_frames_l_i:
            return_frames_l.append(gray.copy())

        if frame_i in return_frames_r_i:
            return_frames_r.append(gray.copy())

        if show:
            frame = aruco.drawDetectedMarkers(frame, corners_list)

        if ids is not None:
            for id, corners in zip(ids, corners_list):
                if id == 0:
                    l.append(np.append(frame_i, np.array(corners[0]).mean(axis=0)))
                if id == 1:
                    r.append(np.append(frame_i, np.array(corners[0]).mean(axis=0)))

            if list(ids) == [0, 1]:
                both.append(np.concatenate([
                    [frame_i], 
                    np.array(corners_list[0][0]).mean(axis=0), 
                    np.array(corners_list[1][0]).mean(axis=0)
                    ]))

        if show:
            cv2.imshow('Test Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        frame_captured, frame = capture.read()
        frame_i += 1

    capture.release()
    cv2.destroyAllWindows()

    l = np.array(l)
    r = np.array(r)
    both = np.array(both)

    if plot:
        plt.plot(l[:,0], -l[:,1], ".", label="left")
        plt.plot(r[:,0], -r[:,1], ".", label="right")
        plt.legend()
        plt.show()
    
    return l, r, both, return_frames_l, return_frames_r


if __name__ == '__main__':
    process_video(sys.argv[1], show=True, plot=True)