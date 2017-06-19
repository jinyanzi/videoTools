#!/usr/bin/python

import cv2
import os
import sys, getopt

##	This script reads the frame index pairs of each video and save image sequence 
##	to the corresponding directory

def extractVideo(vpath, out_dir):
    cap = cv2.VideoCapture(vpath)
  
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print frame_num

    while(True):
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - idx != 1:
            print idx
        d_idx = str(int(idx/100))
        p = os.path.join(out_dir, '0', d_idx)
        if not os.path.exists(p):
            os.makedirs(p)
            print "extracting {0}".format(idx)
        cv2.imwrite(os.path.join(p, "{0}.jpg".format(idx)), frame)
    
    print '%d frames in total' % frame_num

def main(argv):

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'extractVideo.py -i <inputfile> -o <output_dir>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'extractVideo.py -i <inputfile> -o <output_dir>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            vpath = arg
        elif opt in ("-o", "--ofile"):
            out_dir = arg
    print 'Input file is "', vpath
    print 'Output file is "', out_dir

    extractVideo(vpath, out_dir)

if __name__=="__main__":
	main(sys.argv[1:])
