#include "pnp_vision/visualization/drawing.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace pnp_vision::visualization {

const char* status_text(TargetStatus status) {
    switch (status) {
        case TargetStatus::Ok:
            return "OK";
        case TargetStatus::Stale:
            return "STALE";
        case TargetStatus::Fail:
            return "FAIL";
        case TargetStatus::Invalid:
            return "INVALID";
    }
    return "UNKNOWN";
}

cv::Scalar ok_color() {
    return cv::Scalar(100, 220, 90);
}

cv::Scalar warn_color() {
    return cv::Scalar(20, 170, 255);
}

cv::Scalar error_color() {
    return cv::Scalar(70, 70, 255);
}

cv::Scalar text_color() {
    return cv::Scalar(238, 238, 238);
}

cv::Scalar muted_color() {
    return cv::Scalar(175, 175, 175);
}

cv::Scalar panel_bg() {
    return cv::Scalar(24, 24, 24);
}

cv::Scalar panel_border() {
    return cv::Scalar(95, 95, 95);
}

cv::Scalar x_color() {
    return cv::Scalar(0, 0, 255);
}

cv::Scalar y_color() {
    return cv::Scalar(0, 255, 0);
}

cv::Scalar z_color() {
    return cv::Scalar(255, 0, 0);
}

StatusSummary summarize_status(int total, int stale_count, int fail_count) {
    if (fail_count > 0) {
        return {"FAIL", error_color()};
    }
    if (stale_count > 0) {
        return {"STALE", warn_color()};
    }
    if (total > 0) {
        return {"ALL OK", ok_color()};
    }
    return {"NO TARGET", muted_color()};
}

std::string format_angle_deg(double radians, int width) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << std::setw(width)
         << radians * 180.0 / CV_PI;
    return text.str();
}

void draw_detection(
    cv::Mat& frame,
    const detection::Detection& det,
    const std::string& extra) {
    const cv::Rect box(
        cv::Point(
            static_cast<int>(std::lround(det.xyxy[0])),
            static_cast<int>(std::lround(det.xyxy[1]))),
        cv::Point(
            static_cast<int>(std::lround(det.xyxy[2])),
            static_cast<int>(std::lround(det.xyxy[3]))));
    cv::rectangle(frame, box, ok_color(), 2);

    std::ostringstream label;
    label << det.name << " " << std::fixed << std::setprecision(2) << det.conf;
    if (!extra.empty()) {
        label << " " << extra;
    }
    cv::putText(
        frame,
        label.str(),
        cv::Point(box.x, std::max(18, box.y - 6)),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        ok_color(),
        2);
}

void draw_box(
    cv::Mat& frame,
    const detection::Detection& det,
    const cv::Scalar& color) {
    const cv::Rect box(
        cv::Point(
            static_cast<int>(std::lround(det.xyxy[0])),
            static_cast<int>(std::lround(det.xyxy[1]))),
        cv::Point(
            static_cast<int>(std::lround(det.xyxy[2])),
            static_cast<int>(std::lround(det.xyxy[3]))));
    cv::rectangle(frame, box, color, 2);
}

void draw_pnp_axes(
    cv::Mat& frame,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const geometry::PnPPose& pose,
    double axis_length) {
    cv::drawFrameAxes(
        frame,
        camera_matrix,
        dist_coeffs,
        pose.rvec(),
        pose.tvec(),
        static_cast<float>(axis_length),
        2);
}

void draw_translucent_rect(
    cv::Mat& frame,
    int x1,
    int y1,
    int x2,
    int y2,
    const cv::Scalar& color,
    double alpha) {
    const int width = frame.cols;
    const int height = frame.rows;
    x1 = std::clamp(x1, 0, width);
    y1 = std::clamp(y1, 0, height);
    x2 = std::clamp(x2, 0, width);
    y2 = std::clamp(y2, 0, height);
    if (x2 <= x1 || y2 <= y1) {
        return;
    }

    const cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    cv::Mat overlay = frame(roi).clone();
    cv::rectangle(
        overlay,
        cv::Point(0, 0),
        cv::Point(roi.width - 1, roi.height - 1),
        color,
        cv::FILLED);
    cv::addWeighted(overlay, alpha, frame(roi), 1.0 - alpha, 0.0, frame(roi));
}

void draw_fps_overlay(
    cv::Mat& frame,
    double fps,
    double fps_limit) {
    std::ostringstream text;
    text << "FPS " << std::fixed << std::setprecision(1) << fps;
    if (fps_limit > 0.0) {
        text << " / " << std::setprecision(0) << fps_limit;
    }

    constexpr double kFontScale = 0.7;
    constexpr int kThickness = 2;
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        text.str(),
        cv::FONT_HERSHEY_DUPLEX,
        kFontScale,
        kThickness,
        &baseline);
    constexpr int kPad = 10;
    const int panel_width = text_size.width + kPad * 2;
    const int panel_height = text_size.height + kPad * 2;
    const int panel_x = std::max(0, frame.cols - panel_width - 6);
    const int panel_y = 6;

    draw_translucent_rect(
        frame,
        panel_x,
        panel_y,
        panel_x + panel_width,
        panel_y + panel_height,
        panel_bg(),
        0.78);
    cv::rectangle(
        frame,
        cv::Rect(panel_x, panel_y, panel_width, panel_height),
        panel_border(),
        1);
    cv::putText(
        frame,
        text.str(),
        cv::Point(
            panel_x + kPad,
            panel_y + kPad + text_size.height - 4),
        cv::FONT_HERSHEY_DUPLEX,
        kFontScale,
        text_color(),
        kThickness,
        cv::LINE_AA);
}

void draw_target_panel(
    cv::Mat& frame,
    const std::vector<TargetOverlay>& targets,
    int ok_count,
    int stale_count,
    int fail_count) {
    const int height = frame.rows;
    const int width = frame.cols;
    const int panel_width = std::min(540, std::max(280, static_cast<int>(width * 0.72)));
    constexpr int kHeaderHeight = 34;
    constexpr int kRowHeight = 46;
    constexpr int kPad = 8;
    const int total = static_cast<int>(targets.size());
    const int panel_height =
        kHeaderHeight + (total > 0 ? total : 1) * kRowHeight + kPad;
    const int panel_x = std::max(6, width - panel_width - 6);
    const int panel_y = std::max(40, height - panel_height - 8);

    draw_translucent_rect(
        frame,
        panel_x,
        panel_y,
        panel_x + panel_width,
        panel_y + panel_height,
        panel_bg(),
        0.7);
    cv::rectangle(
        frame,
        cv::Rect(panel_x, panel_y, panel_width, panel_height),
        panel_border(),
        1);

    const StatusSummary summary = summarize_status(total, stale_count, fail_count);
    const cv::Scalar header_color = summary.color;
    const char* header_status = summary.text;

    std::ostringstream header;
    header << "TARGETS " << total << " | PnP ";
    if (total > 0) {
        header << ok_count << "/" << total << " OK";
    } else {
        header << "-";
    }
    header << " | " << header_status;
    cv::putText(
        frame,
        header.str(),
        cv::Point(panel_x + 10, panel_y + 23),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        header_color,
        1,
        cv::LINE_AA);

    if (targets.empty()) {
        cv::putText(
            frame,
            "No target in this frame",
            cv::Point(panel_x + 10, panel_y + kHeaderHeight + 28),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            muted_color(),
            1,
            cv::LINE_AA);
        return;
    }

    const int text_x = panel_x + 10;
    for (size_t index = 0; index < targets.size(); ++index) {
        const TargetOverlay& target = targets[index];
        const int row_y = panel_y + kHeaderHeight + static_cast<int>(index) * kRowHeight;
        cv::line(
            frame,
            cv::Point(panel_x + 6, row_y),
            cv::Point(panel_x + panel_width - 6, row_y),
            cv::Scalar(65, 65, 65),
            1);

        std::ostringstream first;
        first << "#" << (index + 1) << " " << target.det.name << " "
              << std::fixed << std::setprecision(2) << target.det.conf;

        std::ostringstream second;
        if (!target.pose.has_value()) {
            first << " | NO POSE | " << status_text(target.status);
            second << "X -- Y -- Z -- | Gimbal R -- P -- Y --";
        } else {
            first << " | Z " << std::fixed << std::setprecision(2)
                  << target.pose->z() << "m | D " << target.pose->distance()
                  << "m | " << status_text(target.status);

            second << "X " << std::fixed << std::setprecision(2)
                   << target.pose->x() << " Y " << target.pose->y()
                   << " Z " << target.pose->z() << " | Gimbal ";
            if (target.gimbal_rpy.has_value()) {
                second << "R " << format_angle_deg(target.gimbal_rpy->val[0])
                       << " P " << format_angle_deg(target.gimbal_rpy->val[1])
                       << " Y " << format_angle_deg(target.gimbal_rpy->val[2]);
            } else {
                second << "R -- P -- Y --";
            }
        }

        cv::putText(
            frame,
            first.str(),
            cv::Point(text_x, row_y + 20),
            cv::FONT_HERSHEY_SIMPLEX,
            0.48,
            target.color,
            1,
            cv::LINE_AA);
        cv::putText(
            frame,
            second.str(),
            cv::Point(text_x, row_y + 38),
            cv::FONT_HERSHEY_SIMPLEX,
            0.48,
            muted_color(),
            1,
            cv::LINE_AA);
    }
}

}  // namespace pnp_vision::visualization
