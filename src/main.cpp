#include <math.h>
#include <stdlib.h>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/gui_widgets.h"

#include "fazeModel.h"

// Force resize to 300, 300 as dlib faces are nearly square

using namespace std;
using namespace dlib;

void log(std::vector<cv::Point> vec) {
	for(int i=0; i<(int)vec.size(); ++i)
		cout<<"("<<vec[i].x<<", "<<vec[i].y<<") ";
	cout<<endl;
}

int main(int argc, char** argv) {
	Faze faze = Faze();
	cv::VideoCapture cap(0);
	cv::Mat frame_clr, frame;
	image_window win, win1;

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("./res/shape_predictor_68_face_landmarks.dat") >> pose_model;

	while(!win.is_closed()) {
		cap >> frame_clr;
		cv::flip(frame_clr, frame_clr, 1);
		cv::cvtColor(frame_clr, frame, CV_BGR2GRAY);

		cv_image<unsigned char> cimg_gray(frame);

		std::vector<rectangle> faces = detector(cimg_gray);

		if((int)faces.size() != 0) {
			rectangle face = faces[0];
			double rx = 300.0/(face.right() - face.left());
			double ry = 300.0/(face.bottom() - face.top());
			rectangle faceResized = rectangle(face.left()*rx, face.top()*ry, face.right()*rx, face.bottom()*ry);
			cv::Rect cvFaceRectResized = cv::Rect(faceResized.left(), faceResized.top(), 
				faceResized.right() - faceResized.left(), faceResized.bottom() - faceResized.top());
			cv::Size sizeNew = cv::Size(frame_clr.cols*rx, frame_clr.rows*ry);
			cv::resize(frame_clr, frame_clr, sizeNew);

			cv_image<bgr_pixel> cimg_clr(frame_clr);

			cv::Mat frameResized;
			cv::resize(frame, frameResized, sizeNew);			
			cv_image<unsigned char> cimg_gray_resized(frameResized);

			cout<<faceResized.right() - faceResized.left()<<", "<<faceResized.bottom() - faceResized.top()<<endl;
			full_object_detection shape = pose_model(cimg_gray_resized, faceResized);
			
			faze.assign(shape, frame_clr);
			faze.setOrigin(faze.ORIGIN_FACE_CENTRE);

			std::vector<double> normal = faze.getNormal();
			cout<<normal[0]<<" "<<normal[1]<<" "<<normal[2]<<endl;
			
			std::vector<cv::Point> mouthCtrsOut = faze.getDescriptors(faze.INDEX_MOUTH_OUTER, faze.DESCRIPTOR_LOCAL);
			std::vector<cv::Point> mouthCtrsIn = faze.getDescriptors(faze.INDEX_MOUTH_INNER, faze.DESCRIPTOR_LOCAL);

			log(mouthCtrsOut);
			log(mouthCtrsIn);

			cv::Mat faceROIRaw = frameResized(cvFaceRectResized), faceROITrans, rotMat(3, 4, CV_64F, 0);
            //faceROIRaw.convertTo(faceROIRaw, CV_64F);
			double cosT = normal[2], sinT = sqrt(1 - normal[2]*normal[2]);
			double p1X, p1Y, p2X, p2Y;
			p1X = 300.0*sinT;
			p1Y = 300.0*cosT;
            p2X = 300.0*cosT + 300.0*sinT;
            p2Y = -300.0*sinT + 300.0*cosT;

            std::vector<cv::Point2f> ptsIn, ptsOut;

            ptsIn.push_back(cv::Point2f(shape.part(0).x()-faceResized.left(), shape.part(0).y()-faceResized.top()));
            int ptX = 2*shape.part(28).x() - shape.part(8).x();
            int ptY = 2*shape.part(28).y() - shape.part(8).y();
            ptsIn.push_back(cv::Point2f(ptX-faceResized.left(), 
            	ptY-faceResized.top()));
            ptsIn.push_back(cv::Point2f(shape.part(16).x()-faceResized.left(),
             shape.part(16).y()-faceResized.top()));
            ptsIn.push_back(cv::Point2f(shape.part(8).x()-faceResized.left(),
             shape.part(8).y()-faceResized.top()));

            ptsOut.push_back(cv::Point2f(0, 150));
            ptsOut.push_back(cv::Point2f(150, 0));
            ptsOut.push_back(cv::Point2f(150, 300));
            ptsOut.push_back(cv::Point2f(300, 150));

            rotMat = cv::getPerspectiveTransform(ptsIn, ptsOut);
            
            cv::Mat v(3, 3, CV_64F, 0);
            std::vector<double> cr(3);
            cr[0] = normal[1];
            cr[1] = -normal[0];
            cr[2] = 0.0;
/*
            v.at<double>(0, 0) = 0;
            v.at<double>(0, 1) = -cr[2];
            v.at<double>(0, 2) = cr[1];
            v.at<double>(1, 0) = cr[2];
            v.at<double>(1, 1) = 0;
            v.at<double>(1, 2) = -cr[0];
            v.at<double>(2, 0) = -cr[1];
            v.at<double>(2, 1) = cr[0];
            v.at<double>(2, 2) = 0;*/

           /* cv::Mat v2 = v*v;

            for(int i=0; i<3; ++i) {
            	for(int j=0; j<3; ++j) {
            		double del = 0;
            		if(i==j) del = 1.0;
            		rotMat.at<double>(i, j) = del + v.at<double>(i, j) + (double)(v2.at<double>(i, j))/(1.0+normal[2]);
            	}
            }*/

            //cv::Rodrigues(normal, rotMat);
            cv::warpPerspective(faceROIRaw, faceROITrans, rotMat, cv::Size(300, 300));
            //cv::imwrite("test.jpg", faceROIRaw);
            faceROITrans.convertTo(faceROITrans, CV_8UC1);
            cout<<shape.part(0).x()-faceResized.left()<<" "<<shape.part(0).y()-faceResized.top()<<endl;
            cout<<ptX-faceResized.left()<<" "<<ptY-faceResized.top()<<endl;
            cout<<shape.part(16).x()-faceResized.left()<<" "<<shape.part(16).y()-faceResized.top()<<endl;
            cout<<shape.part(8).x()-faceResized.left()<<" "<<shape.part(8).y()-faceResized.top()<<endl;

            faceROIRaw.at<uchar>(shape.part(0).x()-faceResized.left(), shape.part(0).y()-faceResized.top()) = 255;
            faceROIRaw.at<uchar>(ptX-faceResized.left(), ptY-faceResized.top()) = 255;
            faceROIRaw.at<uchar>(shape.part(16).x()-faceResized.left(), shape.part(16).y()-faceResized.top()) = 255;
            faceROIRaw.at<uchar>(shape.part(8).x()-faceResized.left(), shape.part(8).y()-faceResized.top()) = 255;
            cv_image<unsigned char> cimg_gray_face(faceROITrans);
            cv_image<unsigned char> cimg_gray_face_raw(faceROIRaw);

            win.clear_overlay();
            win.set_image(cimg_gray_face);
            //win.add_overlay(render_face_detections(shape));
            win1.set_image(cimg_gray_face_raw);
        }
        else {
        	cout<<"No faces"<<endl;
        }
    }
    return 0;
}
