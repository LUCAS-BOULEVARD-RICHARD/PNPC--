#include "pnp_vision/geometry/gimbal.hpp"

#include <cmath>
#include <sstream>

namespace pnp_vision::geometry {

cv::Vec3d calc_gimbal_target_rpy(
    cv::InputArray target_cam,
    double yaw_sign,
    double pitch_sign,
    double min_z,
    std::optional<double> max_range) {
    cv::Mat target = target_cam.getMat().reshape(1, 1);
    if (target.total() != 3) {
        throw GimbalError("target_cam must be 3 elements");
    }
    target.convertTo(target, CV_64F);

    const double xc = target.at<double>(0, 0);
    const double yc = target.at<double>(0, 1);
    const double zc = target.at<double>(0, 2);

    if (!std::isfinite(xc) || !std::isfinite(yc) || !std::isfinite(zc)) {
        throw GimbalError("target_cam contains non-finite values");
    }
    if (zc <= min_z) {
        std::ostringstream message;
        message << "target must be in front of camera: zc=" << zc
                << ", min_z=" << min_z;
        throw GimbalError(message.str());
    }

    if (max_range.has_value()) {
        const double distance = std::sqrt(xc * xc + yc * yc + zc * zc);
        if (distance > *max_range) {
            std::ostringstream message;
            message << "target out of range: distance=" << distance
                    << ", max_range=" << *max_range;
            throw GimbalError(message.str());
        }
    }

    const double gimbal_yaw = yaw_sign * std::atan2(xc, zc);
    const double gimbal_pitch = pitch_sign * -std::atan2(yc, std::hypot(xc, zc));
    return cv::Vec3d(0.0, gimbal_pitch, gimbal_yaw);
}

}  // namespace pnp_vision::geometry
