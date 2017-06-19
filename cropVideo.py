#!/usr/bin/python

import cv2
import os

##	This script reads the frame index pairs of each video and save image sequence 
##	to the corresponding directory

def readIndexFile(filename):
	# file format:
	# videoID	startFrame1	endFrame1	startFrame2 endFrame2
	index_dict = {}
	with open(filename) as f:
		for line in f:
			if not line.startswith('#'):
				tokens = line.rstrip().split('\t')
				idxes = [(int(tokens[1]), int(tokens[2])), (int(tokens[3]), int(tokens[4]))]
				index_dict[tokens[0]] = idxes
	return index_dict


def cropVideo(index_dict, video_dir, out_dir):
	for f in os.listdir(video_dir):
		# get video id from video file name
		#vid = f.split('_', 1)[0]
		vname = f.split('.', 1)[0]
		vpath = os.path.join(video_dir, f)
		key = vname
		if os.path.isfile(vpath) and key in index_dict:
			print key 
			# create directory if not exists
			out_vpath = os.path.join(out_dir, key)
			if not os.path.exists(out_vpath):
				os.makedirs(out_vpath)

			# open video
			cap = cv2.VideoCapture(vpath)
			print index_dict[key]
			for i, seq_idx in enumerate(index_dict[key]):
				# make directory for each sequence
				seq_dir = os.path.join(out_vpath, str(i+1))
				if not os.path.exists(seq_dir):
					os.makedirs(seq_dir)
				# read and write frames
				for frame_num in range(seq_idx[0], seq_idx[1]):
					img_path = os.path.join(seq_dir, (vname + "_" + str(frame_num) + ".png"))
					cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
					ret, frame = cap.read()
					cv2.imwrite(img_path, frame, [0])


def main():
	dict = readIndexFile('/home/jenny/Desktop/dvd/illuminationTest/frame_idx.txt');
	cropVideo(dict, '/home/jenny/Desktop/dvd/videos/IDOT/', '/home/jenny/Desktop/dvd/illuminationTest/imageSeq/')

if __name__=="__main__":
	main()
