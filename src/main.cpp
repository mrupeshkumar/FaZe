#include <math.h>
#include <stdlib.h>
#include <string>

#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/viz.hpp"

#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/gui_widgets.h"

#include "fixedBin.h"
#include "fazeModel.h"
#include "fazeStream.h"

// Force resize to 300, 300 as dlib faces are nearly square

using namespace std;
using namespace dlib;

#define SHOW_DLIB_WINDOW 1
#define SHOW_PLANE_MIMIC 1

void log(std::vector<cv::Point> vec) {
	for(int i=0; i<(int)vec.size(); ++i)
		cout<<"("<<vec[i].x<<", "<<vec[i].y<<") ";
	cout<<endl;
}

void logMat(cv::Mat_<double> mat) {
    cout<<"----------------"<<endl;
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            cout<<mat(i, j)<<" ";
        }
        cout<<endl;
    }
    cout<<"----------------"<<endl;
}

void process(cv::Mat& mat, full_object_detection shape) {

    for(int i=0; i<300; ++i) {
        for(int j=0; j<300; ++j) {
            mat.at<double>(i, j) = 0.0;
        }
    }
    for(int i=0; i<68; ++i) {
        mat.at<double>(shape.part(i).x(), shape.part(i).y()) = 1.0;
    }
}

int main(int argc, char** argv) {
	Faze faze = Faze();
	cv::VideoCapture cap(0);
	cv::Mat frame_clr, frame;

	#if SHOW_DLIB_WINDOW
		image_window win;
	#endif

	#if SHOW_PLANE_MIMIC
		cv::viz::Viz3d vizWin1("Plane face");
	#endif

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("./res/shape_predictor_68_face_landmarks.dat") >> pose_model;

	while(1) {
		cap >> frame_clr;
		cv::flip(frame_clr, frame_clr, 1);
		cv::cvtColor(frame_clr, frame, CV_BGR2GRAY);

		cv_image<unsigned char> cimg_gray(frame);

		std::vector<rectangle> faces = detector(cimg_gray);

		if((int)faces.size() != 0) {
			rectangle face = faces[0];
			cv_image<bgr_pixel> cimg_clr(frame_clr);

			full_object_detection shape = pose_model(cimg_gray, face);

			faze.assign(shape, frame_clr);
			faze.setOrigin(faze.ORIGIN_FACE_CENTRE);

			#if SHOW_DLIB_WINDOW
      	win.clear_overlay();
      	win.set_image(cimg_gray);
				win.add_overlay(render_face_detections(std::vector<full_object_detection>(1, shape)));
			#endif

			#if SHOW_PLANE_MIMIC
				vizWin1.showWidget("Plane Widget", faze.getFacialPlane());
				vizWin1.spinOnce(1, true);
			#endif
    }
  }
  return 0;
}
