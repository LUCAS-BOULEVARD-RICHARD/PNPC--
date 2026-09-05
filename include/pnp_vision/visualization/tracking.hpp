#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <unordered_map>
#include <vector>

#include "pnp_vision/detection/types.hpp"
#include "pnp_vision/geometry/pnp.hpp"
#include "pnp_vision/visualization/drawing.hpp"

namespace pnp_vision::visualization {

constexpr int kTrackCell = 24;
constexpr double kTrackRadius = 48.0;

struct TrackKey {
    int cls = -1;
    int cell_x = 0;
    int cell_y = 0;

    bool operator==(const TrackKey& other) const {
        return cls == other.cls && cell_x == other.cell_x && cell_y == other.cell_y;
    }
};

struct TrackKeyHash {
    size_t operator()(const TrackKey& key) const {
        size_t value = 17;
        value = value * 31 + static_cast<size_t>(key.cls);
        value = value * 31 + static_cast<size_t>(key.cell_x);
        value = value * 31 + static_cast<size_t>(key.cell_y);
        return value;
    }
};

using PoseMap = std::unordered_map<TrackKey, geometry::PnPPose, TrackKeyHash>;

TrackKey detection_key(
    const detection::Detection& det,
    int cell_px = kTrackCell);

std::optional<geometry::PnPPose> match_last_pose(
    const detection::Detection& det,
    const PoseMap& last_pose_by_key,
    TrackKey* matched_key = nullptr,
    int cell_px = kTrackCell,
    double radius_px = kTrackRadius);

void remember_pose(
    const detection::Detection& det,
    const geometry::PnPPose& pose,
    PoseMap& last_pose_by_key,
    const TrackKey* old_key = nullptr,
    int cell_px = kTrackCell);

struct TargetSolveResult {
    std::vector<TargetOverlay> targets;
    int ok_count = 0;
    int stale_count = 0;
    int fail_count = 0;
};

TargetSolveResult solve_targets(
    const std::vector<detection::Detection>& detections,
    const cv::Mat& obj_points,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    PoseMap& last_pose_by_key,
    int cell_px = kTrackCell,
    double radius_px = kTrackRadius);

}  // namespace pnp_vision::visualization
