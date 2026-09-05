#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "pnp_vision/visualization/drawing.hpp"

namespace pnp_vision::visualization {

void draw_top_left_hud(
    cv::Mat& frame,
    const std::vector<TargetOverlay>& targets,
    int ok_count,
    int stale_count,
    int fail_count,
    double fps,
    bool show_fps = true,
    bool show_rpy = true);

}  // namespace pnp_vision::visualization
