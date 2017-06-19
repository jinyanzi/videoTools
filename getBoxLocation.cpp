#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool is_drawing = false;
bool gotBox = false;
bool is_playing = false;
Rect box;

void mouseHandler(int event, int x, int y, int flags, void *param){
	switch( event ){
		case CV_EVENT_MOUSEMOVE:
			if (is_drawing){
				box.width = x-box.x;
				box.height = y-box.y;
			}
			break;
		case CV_EVENT_LBUTTONDOWN:
			is_drawing = true;
			box = Rect( x, y, 0, 0 );
			break;
		case CV_EVENT_LBUTTONUP:
			is_drawing = false;
			if( box.width < 0 ){
				box.x += box.width;
				box.width *= -1;
			}
			if( box.height < 0 ){
				box.y += box.height;
				box.height *= -1;
			}
			gotBox = true;
			is_drawing = false;
			break;
	}
}


void print_help()
{
	cout << "./getBoxLocation <video_path> <output file>" << endl; 
}


int main(int argc, char* argv[] ){

	VideoCapture capture;
	ofstream outfile;
	
	if( argc > 1 ){

		cout << "open " << argv[1] << endl;
		capture.open(argv[1]);
		if( !capture.isOpened() ){
			cout << "Failed to open video " << argv[1] << endl;
		}
		if( argc > 2)
			outfile.open(argv[2]);

		int frame_num = capture.get(CV_CAP_PROP_FRAME_COUNT);
		Mat frame, img;
	
		namedWindow("frame", CV_WINDOW_AUTOSIZE);
		setMouseCallback( "frame", mouseHandler, NULL);

		for( int i = 0; i < frame_num;){
			if( is_playing || i == 0){
				cout << "frame " << i << endl;
				capture >> frame;
				i++;
			}
				
			img = frame.clone();
			if( !is_playing ){
				if(gotBox){
					if( outfile.is_open() )
						outfile << box.x << "\t" << box.y << "\t" 
							<< box.width << "\t" << box.height << endl;
					cout << box.x << "\t" << box.y << "\t" 
						<< box.width << "\t" << box.height << endl;
					gotBox = false;
				}
				rectangle(img, box, Scalar(255,0,0));
				imshow("frame", img);

			}else
				imshow("frame", frame);

			char c = waitKey(30);
			if( c == 27 ){
				break;
			}
			if( c == ' ' ){
				is_playing = !is_playing;
				box = Rect(0,0,0,0);
			}
		}
	}else{
		print_help();
		cout << "please provide video path" << endl;
	}


	if(outfile.is_open())
		outfile.close();
	capture.release();

	return 0;
}

