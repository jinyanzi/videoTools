#! /usr/bin/python

import sys
import numpy as np
import cv2

def visualizeFeatures(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Initiate FAST object with default values
    #fast = cv2.xfeatures2d.FastFeatureDetector()
    sift = cv2.xfeatures2d.SIFT_create()
    #fast = cv2.FastFeatureDetector()

    # Create a mask image for drawing purposes
    #mask = np.zeros_like(old_frame)
   
    if_play = True
    while(1):
        if if_play:
            ret,frame = cap.read()
            
            gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # find and draw the keypoints
            # kp = fast.detect(gray,None)
            kp = sift.detect(gray,None)
            cv2.drawKeypoints(frame, kp, frame, color=(255,0,0))
            #kp = fast.detect(frame,None)
    
        cv2.imshow('frame',frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        if k == ord(' '):
            if_play = not if_play
    
        # Now update the previous frame and previous points
        #old_gray = frame_gray.copy()
        #p0 = good_new.reshape(-1,1,2)
    
    cv2.destroyAllWindows()
    cap.release()

def main():
    if len(sys.argv) > 1:
        visualizeFeatures(sys.argv[1])

if __name__=="__main__":
    main()
