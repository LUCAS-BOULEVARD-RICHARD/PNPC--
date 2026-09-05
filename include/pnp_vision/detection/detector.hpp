#pragma once

#include "pnp_vision/detection/types.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace pnp_vision::detection {

class YoloDetector {
public:
    YoloDetector(
        const std::string& model_path,
        std::vector<std::string> class_names = {},
        double conf_threshold = 0.25,
        double nms_threshold = 0.45,
        int input_size = 640,
        const std::string& runtime = "auto");

    ~YoloDetector();

    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;
    YoloDetector(YoloDetector&&) = delete;
    YoloDetector& operator=(YoloDetector&&) = delete;

    std::vector<Detection> detectRgb(
        const cv::Mat& rgb,
        const std::vector<int>& target_classes = {});

    std::vector<Detection> detectRgb(
        const cv::Mat& rgb,
        double min_conf,
        const std::vector<int>& target_classes = {});

    std::vector<Detection> detectBgr(
        const cv::Mat& bgr,
        const std::vector<int>& target_classes = {});

    std::vector<Detection> detectBgr(
        const cv::Mat& bgr,
        double min_conf,
        const std::vector<int>& target_classes = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pnp_vision::detection
