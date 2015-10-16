#ifndef FAZE_MODEL_H
#define FAZE_MODEL_H

class Faze {

private:

	dlib::full_object_detection faceShape;
	cv::Point origin;
	cv::Mat imageColor, imageGray;

	double yaw, pitch, sigma, symm_x, theta, tau;
	std::vector<double> normal, gaze;

	std::vector<cv::Point> descriptors;
	/*
		0 : INDEX_LEFT_EYE_PUPIL
		1 : INDEX_RIGHT_EYE_PUPIL
	*/

	void computePupil(int mode);
	void computeNormal();
	void computeGaze(int mode);
    std::vector<cv::Point> getIntermediateDescriptors(int index);
    void relativeToOrigin(std::vector<cv::Point>& vec);

    friend class Stream;

public:
	static const int MODE_LEFT = 0;
	static const int MODE_RIGHT = 0;

	static const int MODE_PUPIL_SP = 0;
	static const int MODE_PUPIL_CDF = 1;

	static const int MODE_GAZE_VA = 0;
	static const int MODE_GAZE_QE = 1;

	static const int INDEX_LEFT_EYE = 0;
	static const int INDEX_LEFT_EYEBROW = 1;
	static const int INDEX_RIGHT_EYE = 2;
	static const int INDEX_RIGHT_EYEBROW = 3;
	static const int INDEX_NOSE_UPPER = 4;
	static const int INDEX_NOSE_LOWER = 5;	
	static const int INDEX_MOUTH_OUTER = 6;
	static const int INDEX_MOUTH_INNER = 7;
	static const int INDEX_LEFT_EYE_PUPIL = 8;
	static const int INDEX_RIGHT_EYE_PUPIL = 9;

	static const int ORIGIN_IMAGE = 0;
	static const int ORIGIN_FACE_CENTRE = 1;

	static const int DESCRIPTOR_GLOBAL = 0;
	static const int DESCRIPTOR_LOCAL = 1;

	double Rn;
	double Rm;

	double MAG_NOR;
	double MAG_CR, MAG_LR, MAG_CP, MAG_CM;
	double ALPHA, THETA;

	Faze();

    void assign(dlib::full_object_detection shape, cv::Mat image, int modePupil = MODE_PUPIL_SP, int modeGaze = MODE_GAZE_VA);

    cv::Point getPupil(int mode);
    std::vector<cv::Point> getDescriptors(int index, int mode = DESCRIPTOR_GLOBAL);
    std::vector<double> getGaze();
    std::vector<double> getNormal();
    dlib::full_object_detection getShape();

    void setOrigin(cv::Point origin);
    void setOrigin(int mode);
};

#endif
