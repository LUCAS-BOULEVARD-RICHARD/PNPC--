#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace pnp_vision::detection {

struct Detection {
    cv::Vec4f xyxy;  // 原始图像坐标下的 x1、y1、x2、y2
    int cls = -1;
    float conf = 0.0F;
    std::string name;
};

std::vector<Detection> keep_highest_confidence(std::vector<Detection> detections);

}  // namespace pnp_vision::detection
