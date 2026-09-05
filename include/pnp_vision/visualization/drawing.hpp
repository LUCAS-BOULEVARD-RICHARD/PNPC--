#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <string>
#include <vector>

#include "pnp_vision/detection/types.hpp"
#include "pnp_vision/geometry/pnp.hpp"

namespace pnp_vision::visualization {

enum class TargetStatus {
    Ok,
    Stale,
    Fail,
    Invalid,
};

const char* status_text(TargetStatus status);

// 总体目标状态摘要及对应颜色。优先级为：
// FAIL > STALE > ALL OK > NO TARGET。HUD 与目标面板共用，保证一致。
struct StatusSummary {
    const char* text = "NO TARGET";
    cv::Scalar color;
};

StatusSummary summarize_status(int total, int stale_count, int fail_count);

// 把云台角度（弧度）格式化为右对齐、保留一位小数的角度字符串，
// 例如 "  12.3"。HUD 与目标面板共用。
std::string format_angle_deg(double radians, int width = 6);

struct TargetOverlay {
    detection::Detection det;
    std::optional<geometry::PnPPose> pose;
    TargetStatus status = TargetStatus::Fail;
    cv::Scalar color;
    std::optional<cv::Vec3d> gimbal_rpy;
};

cv::Scalar ok_color();
cv::Scalar warn_color();
cv::Scalar error_color();
cv::Scalar text_color();
cv::Scalar muted_color();
cv::Scalar panel_bg();
cv::Scalar panel_border();
cv::Scalar x_color();
cv::Scalar y_color();
cv::Scalar z_color();

void draw_detection(
    cv::Mat& frame,
    const detection::Detection& det,
    const std::string& extra = "");

void draw_box(
    cv::Mat& frame,
    const detection::Detection& det,
    const cv::Scalar& color);

void draw_pnp_axes(
    cv::Mat& frame,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const geometry::PnPPose& pose,
    double axis_length = 0.15);

void draw_translucent_rect(
    cv::Mat& frame,
    int x1,
    int y1,
    int x2,
    int y2,
    const cv::Scalar& color,
    double alpha = 0.72);

void draw_fps_overlay(
    cv::Mat& frame,
    double fps,
    double fps_limit);

void draw_target_panel(
    cv::Mat& frame,
    const std::vector<TargetOverlay>& targets,
    int ok_count,
    int stale_count,
    int fail_count);

}  // namespace pnp_vision::visualization
