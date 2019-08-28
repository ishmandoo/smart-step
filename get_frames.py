from process_video import process_video
import os
import sys
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    l, r, both, _, _ = process_video(sys.argv[1])
    l, r = -l, -r
    print(both)
    frame_candidates_l_i, _ = find_peaks(both[:,1], height=-1000)
    frame_candidates_r_i, _ = find_peaks(both[:,3], height=-1000)
    _, _, _, l_frames, r_frames = process_video(sys.argv[1], return_frames_l_i=frame_candidates_l_i, return_frames_r_i=frame_candidates_r_i)
    
    plt.plot(both[:,0], both[:,1], ".")
    plt.plot(both[:,0], both[:,3], ".")
    plt.plot(both[frame_candidates_l_i,0], both[frame_candidates_l_i,1], ".")
    plt.plot(both[frame_candidates_r_i,0], both[frame_candidates_r_i,3], ".")
    plt.show()

    for frame in l_frames + r_frames:
        cv2.imshow('Test Frame', frame)
        if cv2.waitKey(0) & 0xFF == ord(' '):
                continue
    