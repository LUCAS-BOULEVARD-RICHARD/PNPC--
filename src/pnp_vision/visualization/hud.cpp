#include "pnp_vision/visualization/hud.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace pnp_vision::visualization {

namespace {

struct Segment {
    std::string text;
    cv::Scalar color;
};

using Line = std::vector<Segment>;

int draw_hud_text(
    cv::Mat& frame,
    const std::string& text,
    cv::Point origin,
    const cv::Scalar& color,
    double font_scale,
    int thickness) {
    cv::putText(
        frame,
        text,
        cv::Point(origin.x + 2, origin.y + 2),
        cv::FONT_HERSHEY_DUPLEX,
        font_scale,
        cv::Scalar(18, 18, 18),
        thickness + 1,
        cv::LINE_AA);
    cv::putText(
        frame,
        text,
        origin,
        cv::FONT_HERSHEY_DUPLEX,
        font_scale,
        color,
        thickness,
        cv::LINE_AA);

    int baseline = 0;
    return cv::getTextSize(
               text,
               cv::FONT_HERSHEY_DUPLEX,
               font_scale,
               thickness,
               &baseline)
        .width;
}

int measure_line(const Line& line, double font_scale, int thickness) {
    int total = 0;
    for (const Segment& segment : line) {
        int baseline = 0;
        total += cv::getTextSize(
                     segment.text,
                     cv::FONT_HERSHEY_DUPLEX,
                     font_scale,
                     thickness,
                     &baseline)
                     .width +
                 12;
    }
    return total;
}

void draw_line(
    cv::Mat& frame,
    const Line& line,
    cv::Point origin,
    double font_scale,
    int thickness) {
    for (const Segment& segment : line) {
        const int width = draw_hud_text(
            frame, segment.text, origin, segment.color, font_scale, thickness);
        origin.x += width + 12;
    }
}

std::vector<Line> build_hud_lines(
    const std::vector<TargetOverlay>& targets,
    int ok_count,
    int stale_count,
    int fail_count,
    double fps,
    bool show_fps,
    bool show_rpy) {
    const int total = static_cast<int>(targets.size());
    const StatusSummary summary = summarize_status(total, stale_count, fail_count);
    const std::string status = summary.text;
    const cv::Scalar status_color = summary.color;

    std::ostringstream status_line;
    if (show_fps) {
        status_line << "FPS " << std::fixed << std::setprecision(1) << fps << " | ";
    }
    status_line << "PnP ";
    if (total > 0) {
        status_line << ok_count << "/" << total << " OK";
    } else {
        status_line << "-";
    }
    status_line << " | " << status;

    std::vector<Line> lines;
    lines.push_back({{status_line.str(), status_color}});
    if (targets.empty()) {
        lines.push_back({{"No target in this frame", muted_color()}});
        return lines;
    }

    for (size_t index = 0; index < targets.size(); ++index) {
        const TargetOverlay& target = targets[index];
        std::ostringstream title;
        title << "#" << (index + 1) << " " << target.det.name << " conf "
              << std::fixed << std::setprecision(2) << target.det.conf
              << " | " << status_text(target.status);
        lines.push_back({{title.str(), target.color}});

        if (!target.pose.has_value()) {
            lines.push_back({{"X -- Y -- Z -- D --", muted_color()}});
            if (show_rpy) {
                lines.push_back({{"Gimbal R -- P -- Y --", muted_color()}});
            }
            continue;
        }

        std::ostringstream xyz;
        xyz << std::fixed << std::setprecision(2);
        xyz << target.pose->x() << "m";
        std::ostringstream y_text;
        y_text << std::fixed << std::setprecision(2) << target.pose->y() << "m";
        std::ostringstream z_text;
        z_text << std::fixed << std::setprecision(2) << target.pose->z() << "m";
        std::ostringstream d_text;
        d_text << std::fixed << std::setprecision(2) << target.pose->distance() << "m";
        lines.push_back({
            {"X ", x_color()},
            {xyz.str(), text_color()},
            {"   Y ", y_color()},
            {y_text.str(), text_color()},
            {"   Z ", z_color()},
            {z_text.str(), text_color()},
            {"   D ", ok_color()},
            {d_text.str(), text_color()},
        });

        if (show_rpy) {
            if (!target.gimbal_rpy.has_value()) {
                lines.push_back({{"Gimbal R -- P -- Y --", muted_color()}});
            } else {
                const std::string groll =
                    format_angle_deg(target.gimbal_rpy->val[0]);
                const std::string gpitch =
                    format_angle_deg(target.gimbal_rpy->val[1]);
                const std::string gyaw =
                    format_angle_deg(target.gimbal_rpy->val[2]);
                lines.push_back({
                    {"Gimbal R ", warn_color()},
                    {groll, text_color()},
                    {"  P ", warn_color()},
                    {gpitch, text_color()},
                    {"  Y ", warn_color()},
                    {gyaw, text_color()},
                });
            }
        }
    }
    return lines;
}

}  // namespace

void draw_top_left_hud(
    cv::Mat& frame,
    const std::vector<TargetOverlay>& targets,
    int ok_count,
    int stale_count,
    int fail_count,
    double fps,
    bool show_fps,
    bool show_rpy) {
    const std::vector<Line> lines =
        build_hud_lines(
            targets,
            ok_count,
            stale_count,
            fail_count,
            fps,
            show_fps,
            show_rpy);
    const int max_available_width = std::max(280, frame.cols - 16);
    constexpr int kPad = 12;
    constexpr int kLineHeight = 36;
    constexpr int kThickness = 2;

    double font_scale = 1.0;
    for (const double candidate : {1.0, 0.9, 0.8, 0.7, 0.6}) {
        int max_width = 0;
        for (const Line& line : lines) {
            max_width = std::max(max_width, measure_line(line, candidate, kThickness));
        }
        if (max_width <= max_available_width) {
            font_scale = candidate;
            break;
        }
    }

    int max_width = 0;
    for (const Line& line : lines) {
        max_width = std::max(max_width, measure_line(line, font_scale, kThickness));
    }
    const int panel_width = std::min(max_available_width, max_width + kPad * 2);
    const int panel_height = kPad * 2 + static_cast<int>(lines.size()) * kLineHeight;
    const int panel_x = 6;
    const int panel_y = 6;

    draw_translucent_rect(
        frame,
        panel_x,
        panel_y,
        panel_x + panel_width,
        panel_y + panel_height,
        panel_bg(),
        0.82);
    cv::rectangle(
        frame,
        cv::Rect(panel_x, panel_y, panel_width, panel_height),
        panel_border(),
        2);

    int text_x = panel_x + kPad;
    int text_y = panel_y + kPad + 26;
    for (const Line& line : lines) {
        draw_line(frame, line, cv::Point(text_x, text_y), font_scale, kThickness);
        text_y += kLineHeight;
    }
}

}  // namespace pnp_vision::visualization
