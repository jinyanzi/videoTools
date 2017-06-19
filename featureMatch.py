#!/usr/bin/python

import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

def getBoundingBoxes(fgmask):
        boxes = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        im2, cnts, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 400:
                continue
                                 
            # compute the bounding box for the contour, draw it on the frame,
            boxes.append(cv2.boundingRect(c))

        return (fgmask, boxes)

def ifContains(box, pt):
    return pt[0] >= box[0] and pt[0] <= box[0]+box[2] and pt[1] >= box[1] and pt[1] <= box[1] + box[3]

# params for ShiTomasi corner detection
def featureMatch(video_path, interval = 1):
	print "Open video ", video_path
	cap = cv2.VideoCapture(video_path)
	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create()
	# BFMatcher with default params
	bf = cv2.BFMatcher()
        fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
        # fgbg = cv2.createBackgroundSubtractorMOG2()

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (4,4),maxLevel = 5,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

        feature_params = dict(maxCorners = 100,qualityLevel = 0.3, minDistance = 7,blockSize = 7)
	# FLANN parameters
	# FLANN_INDEX_KDTREE = 0
	# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	# search_params = dict(checks=50)   # or pass empty dictionary

	# flann = cv2.FlannBasedMatcher(index_params,search_params)

	ret, img1 = cap.read()
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	kp1, des1 = sift.detectAndCompute(img1,None)
        # p = cv2.goodFeaturesToTrack(gray1, mask = None, **feature_params)
        # print p
        fgmask = fgbg.apply(img1)
	# height, width, channels = img1.shape
	
	if_play = True

	while(1):
		if if_play:
			ret, img2 = cap.read()
			if not ret:
				break

                        res = img2.copy()
                        # compute optical flow
                        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                        
                        # background subtraction
                        fgmask = fgbg.apply(img2)
                        fgmask, boxes = getBoundingBoxes(fgmask)
                        
			# find the keypoints and descriptors with SIFT
			kp2, des2 = sift.detectAndCompute(img2,None)
                        
			matches = bf.knnMatch(des1,des2, k=2)
			# matches = flann.knnMatch(des1,des2,k=2)
			# Apply ratio test
			good = []
                        p0 = []
                        p0_match = []
			for m,n in matches:
                                # queryIdx is the indexes in kp1, trainIdx is the index in kp2
                                pt1 = kp1[m.queryIdx].pt
                                pt2 = kp2[n.trainIdx].pt
                                if_check = False
                                for b in boxes:
                                    #if fgmask[pt1[1], pt1[0]] == 0 or fgmask[pt2[1], pt2[0]] == 0:
                                    if ifContains(b, pt1) or ifContains(b, pt2):
                                        if_check = True
                                        break

				if if_check and m.distance < 0.75*n.distance:
				    good.append([m])
                                    p0.append(pt1)
                                    p0_match.append(pt2)
                                    print pt1, pt2
                       
                        if len(p0) > 0:
                            p0 = np.array(p0).astype('float32')
                            p0_match = np.array(p0).astype('float32')
                            row, col = p0.shape
                            p0 = p0.reshape(row, 1, col)
                            p0_match = p0_match.reshape(row, 1, col)

                            p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
                            # Select good points
                            good_new = p1[st==1]
                            good_old = p0[st==1]
                            good_match = p0_match[st==1]

			# cv2.drawMatchesKnn expects list of lists as matches.
			img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
                        for (x, y, w, h) in boxes:
                            cv2.rectangle(res, (x, y), (x+w, y+h), (255, 0, 0))
                        
                        if len(p0) > 0:
                            color = np.random.randint(0,255,(len(good_new),3))
                            for i,(new,old, match) in enumerate(zip(good_new,good_old, good_match)):
                                (x1, y1) = old.ravel()
                                (x2, y2) = new.ravel()
                                (x11, y11) = match.ravel()
                                res = cv2.line(res, (x1, y1), (x2, y2), color[i].tolist(), 1)
                                res = cv2.line(res, (x1, y1), (x11, y11), (0, 255, 0))
                                res = cv2.circle(res, (x1, y1), 2, color[i].tolist(), 1)
                                res = cv2.circle(res, (x11, y11), 2, color[i].tolist(), 1)

			#plt.imshow(img3),plt.show()
			#break
						
		cv2.imshow('frame', img3) 
                cv2.imshow('fgbg', fgmask)
                cv2.imshow('res', res)
		# take keyboard input
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
		if k == ord(' '):
			if_play = not if_play
			   
                img1 = img2
                gray1 = gray2
                kp1 = kp2
                des1 = des2
				
	cap.release()
	cv2.destroyAllWindows()


def main():
	if len(sys.argv) > 1:
		featureMatch(sys.argv[1], 1)

if __name__=="__main__":
	main()
