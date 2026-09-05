#pragma once

#include "pnp_vision/detection/types.hpp"

#include <array>
#include <optional>

namespace pnp_vision::geometry {

class PnPPose {
public:
    PnPPose() = default;
    PnPPose(cv::Mat rvec, cv::Mat tvec);

    const cv::Mat& rvec() const;
    const cv::Mat& tvec() const;

    double x() const;
    double y() const;
    double z() const;
    double distance() const;

private:
    cv::Mat rvec_;
    cv::Mat tvec_;
};

cv::Mat object_points(double width_m, double height_m);

std::optional<PnPPose> solve_pnp_pose(
    const detection::Detection& det,
    const cv::Mat& obj_points,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs);

std::optional<std::array<double, 4>> solve_pnp(
    const detection::Detection& det,
    const cv::Mat& obj_points,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs);

cv::Mat rvec_to_rotation_matrix(const cv::Mat& rvec);

cv::Vec3d rotation_matrix_to_euler_zyx(
    const cv::Mat& rotation_matrix,
    bool degrees = true);

cv::Vec3d rvec_to_euler_zyx(
    const cv::Mat& rvec,
    bool degrees = true);

}  // namespace pnp_vision::geometry
