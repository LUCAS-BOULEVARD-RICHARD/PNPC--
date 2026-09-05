#pragma once

#include <optional>
#include <string>
#include <vector>

namespace pnp_vision::config {

#ifdef PNP_USE_TENSORRT
inline constexpr const char* kDefaultModelPath =
    "model/labubu_yolov8n.engine";
#else
inline constexpr const char* kDefaultModelPath =
    "model/labubu_yolov8n.onnx";
#endif

inline constexpr const char* kDefaultConfigPath = "config/config.yaml";

struct CameraSettings {
    std::string mode = "laptop";
    std::string source = "0";
    int read_max_failures = 30;
    int read_retry_delay_ms = 50;
};

struct ExposureSettings {
    double min_us = 8000.0;
    double max_us = 50000.0;
    double max_gain = 100.0;
    double gain_step_coarse = 0.25;
    double gain_step_fine = 0.05;
    double exposure_step_coarse = 1.5;
    double exposure_step_fine = 1.1;
};

struct DetectionSettings {
    std::string model_path = kDefaultModelPath;
    std::string runtime = "auto";
    int input_size = 640;
    double confidence = 0.25;
    double nms = 0.45;
    std::vector<std::string> class_names = {"labubu", "toy"};
    std::vector<int> target_classes;
};

struct ObjectSettings {
    double width_m = 0.30;
    double height_m = 0.30;
};

struct PnpSettings {
    double axis_length_m = 0.15;
    double yaw_sign = 1.0;
    double pitch_sign = 1.0;
    double min_z_m = 0.01;
    std::optional<double> max_range_m;
};

struct TrackingSettings {
    int cell_px = 24;
    double radius_px = 48.0;
};

struct CalibrationSettings {
    std::string pattern = "5x8";
    double square_m = 0.024;
    int frames = 25;
    std::string output = "calibration.yaml";
};

struct WindowSettings {
    bool show = true;
    bool show_fps = true;
    bool show_rpy = true;
    bool show_target_panel = true;
    double fps_limit = 40.0;
};

struct AppConfig {
    CameraSettings camera;
    ExposureSettings exposure;
    std::string calibration_file = "calibration.yaml";
    DetectionSettings detection;
    ObjectSettings object;
    PnpSettings pnp;
    TrackingSettings tracking;
    CalibrationSettings calibration;
    WindowSettings window;
};

AppConfig default_config();
AppConfig load_config(const std::string& path);

}  // namespace pnp_vision::config
