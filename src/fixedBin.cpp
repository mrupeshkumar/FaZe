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

#include "fixedBin.h"
#include "fazeModel.h"
#include "fazeStream.h"
#include "util.h"
#include "pupilDetectionCDF.h"
#include "pupilDetectionSP.h"

/* Code contained in fixedBin.h */