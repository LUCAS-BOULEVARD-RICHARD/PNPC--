#include "pnp_vision/geometry/pnp.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

double vec_value(const cv::Mat& value, int index) {
    return value.reshape(1, 1).at<double>(0, index);
}

}  // namespace

namespace pnp_vision::geometry {

PnPPose::PnPPose(cv::Mat rvec, cv::Mat tvec) {
    if (rvec.total() != 3 || tvec.total() != 3) {
        throw std::invalid_argument("rvec and tvec must contain 3 elements");
    }
    rvec.convertTo(rvec_, CV_64F);
    tvec.convertTo(tvec_, CV_64F);
}

const cv::Mat& PnPPose::rvec() const {
    return rvec_;
}

const cv::Mat& PnPPose::tvec() const {
    return tvec_;
}

double PnPPose::x() const {
    return vec_value(tvec_, 0);
}

double PnPPose::y() const {
    return vec_value(tvec_, 1);
}

double PnPPose::z() const {
    return vec_value(tvec_, 2);
}

double PnPPose::distance() const {
    return cv::norm(tvec_, cv::NORM_L2);
}

cv::Mat object_points(double width_m, double height_m) {
    const double w2 = width_m / 2.0;
    const double h2 = height_m / 2.0;

    cv::Mat points(4, 3, CV_32F);
    points.at<float>(0, 0) = static_cast<float>(-w2);
    points.at<float>(0, 1) = static_cast<float>(-h2);
    points.at<float>(0, 2) = 0.0F;

    points.at<float>(1, 0) = static_cast<float>(w2);
    points.at<float>(1, 1) = static_cast<float>(-h2);
    points.at<float>(1, 2) = 0.0F;

    points.at<float>(2, 0) = static_cast<float>(w2);
    points.at<float>(2, 1) = static_cast<float>(h2);
    points.at<float>(2, 2) = 0.0F;

    points.at<float>(3, 0) = static_cast<float>(-w2);
    points.at<float>(3, 1) = static_cast<float>(h2);
    points.at<float>(3, 2) = 0.0F;

    return points;
}

std::optional<PnPPose> solve_pnp_pose(
    const detection::Detection& det,
    const cv::Mat& obj_points,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs) {
    cv::Mat image_points(4, 2, CV_32F);
    image_points.at<float>(0, 0) = det.xyxy[0];
    image_points.at<float>(0, 1) = det.xyxy[1];
    image_points.at<float>(1, 0) = det.xyxy[2];
    image_points.at<float>(1, 1) = det.xyxy[1];
    image_points.at<float>(2, 0) = det.xyxy[2];
    image_points.at<float>(2, 1) = det.xyxy[3];
    image_points.at<float>(3, 0) = det.xyxy[0];
    image_points.at<float>(3, 1) = det.xyxy[3];

    cv::Mat rvec;
    cv::Mat tvec;
    bool ok = false;
    try {
        ok = cv::solvePnP(
            obj_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_ITERATIVE);
    } catch (const cv::Exception&) {
        return std::nullopt;
    }

    if (!ok) {
        return std::nullopt;
    }
    return PnPPose(std::move(rvec), std::move(tvec));
}

std::optional<std::array<double, 4>> solve_pnp(
    const detection::Detection& det,
    const cv::Mat& obj_points,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs) {
    const std::optional<PnPPose> pose =
        solve_pnp_pose(det, obj_points, camera_matrix, dist_coeffs);
    if (!pose.has_value()) {
        return std::nullopt;
    }
    return std::array<double, 4>{pose->x(), pose->y(), pose->z(), pose->distance()};
}

cv::Mat rvec_to_rotation_matrix(const cv::Mat& rvec) {
    if (rvec.total() != 3) {
        throw std::invalid_argument("rvec must contain 3 elements");
    }

    cv::Mat rvec64;
    rvec.convertTo(rvec64, CV_64F);
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec64.reshape(1, 1), rotation_matrix);
    return rotation_matrix;
}

cv::Vec3d rotation_matrix_to_euler_zyx(
    const cv::Mat& rotation_matrix,
    bool degrees) {
    if (rotation_matrix.rows != 3 || rotation_matrix.cols != 3) {
        throw std::invalid_argument("rotation matrix must be 3x3");
    }

    cv::Mat r;
    rotation_matrix.convertTo(r, CV_64F);

    const double r00 = r.at<double>(0, 0);
    const double r10 = r.at<double>(1, 0);
    const double r20 = r.at<double>(2, 0);
    const double r21 = r.at<double>(2, 1);
    const double r22 = r.at<double>(2, 2);
    const double r12 = r.at<double>(1, 2);
    const double r11 = r.at<double>(1, 1);

    const double sy = std::hypot(r00, r10);
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    if (sy < 1e-6) {
        roll = std::atan2(-r12, r11);
        pitch = std::atan2(-r20, sy);
        yaw = 0.0;
    } else {
        roll = std::atan2(r21, r22);
        pitch = std::atan2(-r20, sy);
        yaw = std::atan2(r10, r00);
    }

    if (degrees) {
        roll *= 180.0 / CV_PI;
        pitch *= 180.0 / CV_PI;
        yaw *= 180.0 / CV_PI;
    }
    return cv::Vec3d(roll, pitch, yaw);
}

cv::Vec3d rvec_to_euler_zyx(const cv::Mat& rvec, bool degrees) {
    return rotation_matrix_to_euler_zyx(rvec_to_rotation_matrix(rvec), degrees);
}

}  // namespace pnp_vision::geometry
