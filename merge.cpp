#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define VIDEO_PER_ROW 2

void print_help()
{
	cout << "./convert <input_video_num> <video_path_1> ... <video_path_n> <output_video_name>" << endl; 
}


int main(int argc, char* argv[] ){
	
    if( argc >= 4 ){

        int video_num = atoi(argv[1]);
        
        VideoCapture *capture = new VideoCapture[video_num];
        Mat big_frame, sub_frame, frame;
        VideoWriter writer;

		int frames = INT_MAX;

        // open videos 
        for( int i = 0; i < video_num; i++){
            cout << "open " << argv[i+2] << endl;
            capture[i].open(argv[i+2]);
            if( !capture[i].isOpened() ){
                cout << "Failed to open video " << argv[i+2] << endl;
                return -1;
            }
            frames = std::min((int)capture[i].get(CV_CAP_PROP_FRAME_COUNT), frames);
        }

        int rows = video_num/VIDEO_PER_ROW;
        if(video_num % VIDEO_PER_ROW != 0)  rows++;


        // assume the input videos have same size and framerate
		int width = capture[0].get(CV_CAP_PROP_FRAME_WIDTH);
		int height = capture[0].get(CV_CAP_PROP_FRAME_HEIGHT);
		double framerate = capture[0].get(CV_CAP_PROP_FPS);

        big_frame.create(height*rows, width*VIDEO_PER_ROW, CV_8UC3);
		writer.open(argv[argc-1], CV_FOURCC('D', 'I', 'V', 'X'), 
				framerate, Size(width, height), true);

		if(!writer.isOpened()){
			cout << "Failed to open video writer for video " << argv[argc-1] << endl;
			return -1;
		}
	
		for( int i = 0; i < frames;){
			cout << "frame " << i << endl;
            for( int j = 0; j < video_num; j++){
			    capture[j] >> frame;
                if( frame.channels() == 1 )
                    cvtColor(frame, frame, CV_GRAY2BGR );
                int start_row = (j/VIDEO_PER_ROW) * height;
                int end_row = start_row + height;
                int start_col =  (j%VIDEO_PER_ROW) * width;
                int end_col = start_col + width;
                cout << "row " << start_row << "~" << end_row << " cols " << start_col << "~" << end_col << endl;
                sub_frame = big_frame( Range(start_row, end_row), Range(start_col, end_col) );
                frame.copyTo(sub_frame);
                if(j == 0)
                    imshow("small", frame);
            }
            imshow("big", big_frame);
			writer.write(big_frame);
			i++;
		}

        for(int i = 0; i < video_num; i++ ){
            capture[i].release();
        }
        delete[] capture;

	}else{
		print_help();
	}

	return 0;
}

