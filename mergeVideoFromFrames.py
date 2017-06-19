#!/usr/bin/python

import cv2
import os
import sys, getopt

def listAndSortDir(dir):
    return sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0]))


def mergeVideo(img_dir, out_path):

    writer = None
    for top_dir in listAndSortDir(img_dir):
        top_dir_path = os.path.join(img_dir, top_dir)
        for dir in listAndSortDir(top_dir_path):
            dir_path = os.path.join(top_dir_path, dir)
            print dir_path
            for img in listAndSortDir(dir_path):
                img_path = os.path.join(dir_path, img)
                print img_path
                frame = cv2.imread(img_path)
                if not writer:
                    height, width, channels = frame.shape
                    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
                    writer = cv2.VideoWriter(out_path, fourcc, 30, (width, height), 1)
                writer.write(frame)


def main(argv):

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'extractVideo.py -i <input_dir> -o <output_video>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'extractVideo.py -i <input_dir> -o <output_video>'
            sys.exit()
        elif opt in ("-i", "--idir"):
            img_dir = arg
        elif opt in ("-o", "--ofile"):
            out_path = arg
    print 'Input frame directory is "', img_dir
    print 'Output file is "', out_path

    mergeVideo(img_dir, out_path)

if __name__=="__main__":
	main(sys.argv[1:])
