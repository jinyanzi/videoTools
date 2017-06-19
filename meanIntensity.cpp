#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){

	if(argc < 2){
		cerr << "Please provide the path of the video" << endl;
		return -1;
	}

	VideoCapture cap(argv[1]);
	if(!cap.isOpened()){
		cerr << "Failed to open video " << argv[1] << endl;
		return -1;
	}

	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int frame_num = cap.get(CV_CAP_PROP_FRAME_COUNT);
	cout << "Open video " << argv[1] << "\n[" << width << "x" << height << "]\t" << frame_num << " frames in total." << endl;

	ofstream outfile;
	outfile.open(argc < 3 ? "intensities.txt" : argv[2]);
	cout << "write output to file " << argv[2] << endl;
	for(int i = 1; i <= frame_num; ++i){
		Mat frame, gray, left, right;
		cap >> frame;
		if(frame.empty())	continue;

		cvtColor(frame, gray, COLOR_BGR2GRAY);
		left = gray(Rect(0, 0, width/2, height));
		right = gray(Rect(width/2, 0, width/2, height));

		double mean_left = mean(left)[0];
		double mean_right = mean(right)[0];
		cout << "frame " << i << "/" << frame_num << endl;
		outfile << i << "\t" << mean_left << "\t" << mean_right << endl;
		//imshow("entire", frame);
		//imshow("left", left);
		//imshow("right", right);
		//if(waitKey(30) == 'q')	break;
	}

	outfile.close();

	return 0;
}
