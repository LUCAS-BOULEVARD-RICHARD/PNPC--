#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace pnp_vision::geometry {

struct CalibrationData {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
};

CalibrationData load_calibration(const std::string& path);

}  // namespace pnp_vision::geometry
