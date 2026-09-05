#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <stdexcept>

namespace pnp_vision::geometry {

class GimbalError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

cv::Vec3d calc_gimbal_target_rpy(
    cv::InputArray target_cam,
    double yaw_sign = 1.0,
    double pitch_sign = 1.0,
    double min_z = 0.01,
    std::optional<double> max_range = std::nullopt);

}  // namespace pnp_vision::geometry
