#!/usr/bin/python

import sys
import cv2
import numpy as np


# params for ShiTomasi corner detection
def visualizeOF(video_path, interval = 1, outpath = None, resize_factor = 1.0):
	print "Open video ", video_path
	cap = cv2.VideoCapture(video_path)
	step_size = 6

        #cap.set(cv2.CAP_PROP_POS_FRAMES, 3000)

	ret, frame1 = cap.read()
	ref_frame = frame1
	prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	hsv = np.zeros_like(frame1)
	hsv[...,1] = 255
	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	height, width, channels = frame1.shape
	x = np.arange(3, width, step = step_size)
	y = np.arange(3, height, step = step_size)
	accumulation = np.zeros((height, width, 2), dtype=np.float64)

        print height, width
        if outpath:
            #big_frame = np.zeros((height, width*2, 3), dtype=np.uint8)
            resized_frame = np.zeros((int(height*resize_factor), int(width*resize_factor), 3), dtype=np.uint8)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            of_out = cv2.VideoWriter(outpath,fourcc, 30.0, (int(width*resize_factor), int(height*resize_factor)))

        # params for ShiTomasi corner detection
	feature_params = dict( maxCorners = 10, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )

	if_play = False
	prvs2 = prvs
	c = 1
	last_idx=1

	#while(last_idx < 3300):
        while(1):
		if if_play or outpath:
                        #cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES)+5)
			print cap.get(cv2.CAP_PROP_POS_FRAMES)
			ret, frame = cap.read()
			if not ret:
			    break
                        
                        print(last_idx)
                        # compute optical flow between consecutive frames
                        resized_frame = cv2.resize(frame, (0, 0), fx = resize_factor, fy = resize_factor)
			current = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			p0 = cv2.goodFeaturesToTrack(current, mask = None, **feature_params)
			flow = cv2.calcOpticalFlowFarneback(prvs,current, None, 0.5, 3, 15, 3, 5, 1.2, 0) 
                        
			# compute movement of each pixel
			# draw arrows for every pixel neighbor
			# dist = np.linalg.norm(flow, axis = 2)
			# print np.amax(dist)
			mean_dist_x = cv2.blur(flow[...,0], (step_size, step_size))
			mean_dist_y = cv2.blur(flow[...,1], (step_size, step_size))
			accumulation[...,0] = accumulation[...,0] + mean_dist_x;
			accumulation[...,1] = accumulation[...,1] + mean_dist_y;
                        
            # compute optical flow with larger interval
			if interval > 1 and c == interval:
				flow = cv2.calcOpticalFlowFarneback(prvs2,current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
				mean_dist_x = cv2.blur(flow[...,0], (step_size, step_size))
				mean_dist_y = cv2.blur(flow[...,1], (step_size, step_size))
			
			if interval == 1 or c >= interval:

			    # color board
			    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
			    hsv[...,0] = ang*180/np.pi/2
			    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
			    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

			    for i in x:
			        for j in y:
			    	    #cv2.line(frame1, (i, j), (int(i + accumulation[j, i, 0]), int(j + accumulation[j, i, 1])), (255, 0, 255))
                        # draw optical flow with larger interval
			    	    #cv2.line(frame1, (i, j), (int(i + mean_dist_x[j, i]), int(j + mean_dist_y[j, i])), (0, 255, 0))
                                    cv2.line(resized_frame, (int(i*resize_factor), int(j*resize_factor)), (int((i + mean_dist_x[j, i])*resize_factor), int((j + mean_dist_y[j, i])*resize_factor)), color = (int(bgr[j, i, 0]), int(bgr[j, i, 1]), int(bgr[j, i, 2])), thickness = 2)
			    
				# reset
				c = 1
				accumulation = np.zeros((height, width, 2), dtype=np.float64)
				prvs2 = current
				ref_frame = frame
				last_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
			else:
				c = c+1
                        

        # if interval == 1, c is reset
		if interval == 1 or c == 1:
	            #cv2.imshow('optical flow',bgr)
		    #cv2.imshow('frame', frame1)
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) == 3201:
                        cv2.imwrite('flow-3201.jpg', bgr)
                        cv2.imwrite('frame-3201.jpg', frame)
                        cv2.imwrite('colorBoard-3201.jpg', frame1)
                    if outpath:
                        # big_frame[:, 0:width, :] = frame1
                        # big_frame[:, width:width*2, :] = bgr
                        # of_out.write(big_frame)
                        of_out.write(resized_frame)
            # take keyboard input
		    k = cv2.waitKey(30) & 0xff
		    if k == 27:
		    	break
		    if k == ord(' '):
		    	if_play = not if_play
		    if k == ord('s'):
		    	cv2.imwrite('opticalfb.png',frame1)
		    	cv2.imwrite('opticalhsv.png',bgr)
           
		if if_play or outpath:
			prvs = current    
			frame1 = ref_frame
                
	cap.release()
        if outpath:
            of_out.release()
	cv2.destroyAllWindows()

def main():
    if len(sys.argv) > 3:
        visualizeOF(sys.argv[1], 1, sys.argv[2], float(sys.argv[3]))
    elif len(sys.argv) > 2:
        visualizeOF(sys.argv[1], 1, sys.argv[2])
    elif len(sys.argv) > 1:
        visualizeOF(sys.argv[1], 30)
    else:
        print "no argument"

if __name__=="__main__":
    main()
