#pragma once

#include "pnp_vision/config/config.hpp"

#include <optional>
#include <string>
#include <vector>

namespace pnp_vision::tasks {

enum class TaskKind {
    Detection,
    Distance,
    Pose,
    Visualize,
    Calibration,
    Exposure,
};

struct TaskArgs {
    TaskKind kind = TaskKind::Detection;

    std::string camera;
    std::string source;
    std::string model;
    std::string runtime;
    std::string calib;
    config::ExposureSettings exposure;
    config::TrackingSettings tracking;
    int camera_max_failures = 30;
    int camera_retry_delay_ms = 50;

    bool has_object_width = false;
    bool has_object_height = false;
    double object_width = 0.0;
    double object_height = 0.0;
    double axis_length = 0.0;

    double conf = 0.0;
    double nms = 0.0;
    int imgsz = 0;
    std::vector<int> classes;
    std::vector<std::string> class_names;

    bool no_show = false;
    bool show_fps = false;
    bool show_rpy = false;
    bool show_target_panel = false;
    double fps_limit = 0.0;
    double yaw_sign = 1.0;
    double pitch_sign = 1.0;
    double min_z = 0.0;
    std::optional<double> max_range;

    std::string pattern;
    double square = 0.0;
    int count = 0;
    std::string output;

    TaskArgs();
};

TaskArgs parse_task_args(int argc, char** argv, TaskKind kind);

void print_usage(TaskKind kind);

}  // namespace pnp_vision::tasks
