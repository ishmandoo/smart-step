from process_video import process_video
import os
import sys
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import cv2

def get_frames(path, show=False):
    positions, _ = process_video(path)

    frame_candidates_l, _ = find_peaks(-positions[:,1], height=-1000)
    frame_candidates_r, _ = find_peaks(-positions[:,3], height=-1000)

    frame_candidates_l_i = positions[frame_candidates_l.astype(int),0].astype(int)
    frame_candidates_r_i = positions[frame_candidates_r.astype(int),0].astype(int)

    _, frames = process_video(sys.argv[1], return_frames_i=sorted(list(frame_candidates_l_i)+list(frame_candidates_r_i)))
    
    if show:
        plt.plot(positions[:,0], positions[:,1], ".")
        plt.plot(positions[:,0], positions[:,3], ".")
        plt.plot(positions[frame_candidates_l,0], positions[frame_candidates_l,1], ".")
        plt.plot(positions[frame_candidates_r,0], positions[frame_candidates_r,3], ".")
        plt.show()

        for frame in frames:
            cv2.imshow('Test Frame', frame)
            if cv2.waitKey(0) & 0xFF == ord(' '):
                    continue
    return frames
    

if __name__ == '__main__':
    get_frames(sys.argv[1], show=True)