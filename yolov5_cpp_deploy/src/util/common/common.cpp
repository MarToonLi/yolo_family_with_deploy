#include <opencv2/opencv.hpp>
#include "common.h"

namespace wikky_algo
{
    void ColorManager::initColorMap() {
        this->colorMap["black"] = cv::Scalar(0, 0, 0);
        this->colorMap["red"] = cv::Scalar(0, 0, 255);         // ng
        this->colorMap["blue"] = cv::Scalar(255, 0, 0);        
        this->colorMap["green"] = cv::Scalar(0, 255, 0);       // ok
        this->colorMap["yellow"] = cv::Scalar(0, 255, 255);    
        this->colorMap["white"] = cv::Scalar(255, 255, 255);
        this->colorMap["purple"] = cv::Scalar(210, 130, 150);
        this->colorMap["gray"] = cv::Scalar(128, 128, 128);    // mask
        this->colorMap["null"] = cv::Scalar(13, 206, 255);
    }
}

