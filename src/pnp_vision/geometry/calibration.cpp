#include "pnp_vision/geometry/calibration.hpp"

#include <stdexcept>

namespace pnp_vision::geometry {

CalibrationData load_calibration(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("calibration file not found: " + path);
    }

    CalibrationData data;
    fs["camera_matrix"] >> data.camera_matrix;
    fs["dist_coeffs"] >> data.dist_coeffs;
    fs.release();

    if (data.camera_matrix.empty() || data.dist_coeffs.empty()) {
        throw std::invalid_argument("invalid calibration file: " + path);
    }
    return data;
}

}  // namespace pnp_vision::geometry
