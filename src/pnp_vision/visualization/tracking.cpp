#include "pnp_vision/visualization/tracking.hpp"

#include <cmath>

namespace pnp_vision::visualization {

TrackKey detection_key(
    const detection::Detection& det,
    int cell_px) {
    const float cx = (det.xyxy[0] + det.xyxy[2]) * 0.5F;
    const float cy = (det.xyxy[1] + det.xyxy[3]) * 0.5F;
    return TrackKey{
        det.cls,
        static_cast<int>(std::lround(cx / cell_px)),
        static_cast<int>(std::lround(cy / cell_px))};
}

std::optional<geometry::PnPPose> match_last_pose(
    const detection::Detection& det,
    const PoseMap& last_pose_by_key,
    TrackKey* matched_key,
    int cell_px,
    double radius_px) {
    const TrackKey exact_key = detection_key(det, cell_px);
    const auto exact = last_pose_by_key.find(exact_key);
    if (exact != last_pose_by_key.end()) {
        if (matched_key != nullptr) {
            *matched_key = exact_key;
        }
        return exact->second;
    }

    const float cx = (det.xyxy[0] + det.xyxy[2]) * 0.5F;
    const float cy = (det.xyxy[1] + det.xyxy[3]) * 0.5F;
    const float radius_sq =
        static_cast<float>(radius_px * radius_px);
    TrackKey best_key{};
    std::optional<geometry::PnPPose> best_pose;
    float best_dist = radius_sq;

    for (const auto& item : last_pose_by_key) {
        const TrackKey& stored_key = item.first;
        if (stored_key.cls != det.cls) {
            continue;
        }
        const float sx = stored_key.cell_x * cell_px;
        const float sy = stored_key.cell_y * cell_px;
        const float dist2 = (cx - sx) * (cx - sx) + (cy - sy) * (cy - sy);
        if (dist2 <= best_dist) {
            best_dist = dist2;
            best_key = stored_key;
            best_pose = item.second;
        }
    }

    if (best_pose.has_value() && matched_key != nullptr) {
        *matched_key = best_key;
    }
    return best_pose;
}

void remember_pose(
    const detection::Detection& det,
    const geometry::PnPPose& pose,
    PoseMap& last_pose_by_key,
    const TrackKey* old_key,
    int cell_px) {
    const TrackKey new_key = detection_key(det, cell_px);
    if (old_key != nullptr && !(*old_key == new_key)) {
        last_pose_by_key.erase(*old_key);
    }
    last_pose_by_key[new_key] = pose;
}

TargetSolveResult solve_targets(
    const std::vector<detection::Detection>& detections,
    const cv::Mat& obj_points,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    PoseMap& last_pose_by_key,
    int cell_px,
    double radius_px) {
    TargetSolveResult result;
    for (const detection::Detection& det : detections) {
        std::optional<geometry::PnPPose> pose =
            geometry::solve_pnp_pose(det, obj_points, camera_matrix, dist_coeffs);
        if (pose.has_value()) {
            TrackKey old_key{};
            match_last_pose(
                det,
                last_pose_by_key,
                &old_key,
                cell_px,
                radius_px);
            remember_pose(
                det,
                *pose,
                last_pose_by_key,
                &old_key,
                cell_px);
            result.targets.push_back(TargetOverlay{
                det,
                pose,
                TargetStatus::Ok,
                ok_color(),
                std::nullopt});
            ++result.ok_count;
        } else {
            TrackKey matched_key{};
            const std::optional<geometry::PnPPose> previous =
                match_last_pose(
                    det,
                    last_pose_by_key,
                    &matched_key,
                    cell_px,
                    radius_px);
            if (previous.has_value()) {
                result.targets.push_back(TargetOverlay{
                    det,
                    previous,
                    TargetStatus::Stale,
                    warn_color(),
                    std::nullopt});
                ++result.stale_count;
            } else {
                result.targets.push_back(TargetOverlay{
                    det,
                    std::nullopt,
                    TargetStatus::Fail,
                    error_color(),
                    std::nullopt});
                ++result.fail_count;
            }
        }
    }
    return result;
}

}  // namespace pnp_vision::visualization
