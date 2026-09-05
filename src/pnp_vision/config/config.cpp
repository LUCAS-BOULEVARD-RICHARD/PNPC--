#include "pnp_vision/config/config.hpp"
#include "pnp_vision/util/text.hpp"
#include "yaml.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace pnp_vision::config {

namespace {

void require_positive(double value, const std::string& name) {
    if (!(value > 0.0)) {
        throw std::invalid_argument(name + " must be positive");
    }
}

void require_nonnegative(double value, const std::string& name) {
    if (!(value >= 0.0)) {
        throw std::invalid_argument(name + " must be non-negative");
    }
}

}  // namespace

AppConfig default_config() {
    return AppConfig{};
}

AppConfig load_config(const std::string& path) {
    ConfigManager& yaml = ConfigManager::getInstance();
    try {
        yaml.init(path);
    } catch (const std::exception& error) {
        throw std::invalid_argument(
            "Failed to parse config " + path + ": " + error.what());
    }

    const auto resolve_path = [&](std::string value) {
        std::filesystem::path resolved(value);
        if (resolved.is_relative()) {
            resolved = std::filesystem::absolute(path).parent_path() / resolved;
        }
        return resolved.lexically_normal().string();
    };

    AppConfig cfg = default_config();

    cfg.camera.mode = yaml.get("camera.mode", cfg.camera.mode);
    cfg.camera.source = yaml.get("camera.source", cfg.camera.source);
    cfg.camera.read_max_failures =
        yaml.get("camera.read_max_failures", cfg.camera.read_max_failures);
    cfg.camera.read_retry_delay_ms =
        yaml.get(
            "camera.read_retry_delay_ms",
            cfg.camera.read_retry_delay_ms);

    cfg.exposure.min_us =
        yaml.get("exposure.min_us", cfg.exposure.min_us);
    cfg.exposure.max_us =
        yaml.get("exposure.max_us", cfg.exposure.max_us);
    cfg.exposure.max_gain =
        yaml.get("exposure.max_gain", cfg.exposure.max_gain);
    cfg.exposure.gain_step_coarse =
        yaml.get("exposure.gain_step_coarse", cfg.exposure.gain_step_coarse);
    cfg.exposure.gain_step_fine =
        yaml.get("exposure.gain_step_fine", cfg.exposure.gain_step_fine);
    cfg.exposure.exposure_step_coarse =
        yaml.get(
            "exposure.exposure_step_coarse",
            cfg.exposure.exposure_step_coarse);
    cfg.exposure.exposure_step_fine =
        yaml.get(
            "exposure.exposure_step_fine",
            cfg.exposure.exposure_step_fine);

    cfg.detection.model_path =
        yaml.get("detection.model_path", cfg.detection.model_path);
    cfg.detection.model_path = resolve_path(cfg.detection.model_path);
    cfg.detection.runtime =
        yaml.get("detection.runtime", cfg.detection.runtime);
    cfg.detection.input_size =
        yaml.get("detection.input_size", cfg.detection.input_size);
    cfg.detection.confidence =
        yaml.get("detection.confidence", cfg.detection.confidence);
    cfg.detection.nms =
        yaml.get("detection.nms", cfg.detection.nms);
    cfg.detection.class_names =
        yaml.getVector<std::string>("detection.class_names");
    if (cfg.detection.class_names.empty()) {
        cfg.detection.class_names = DetectionSettings{}.class_names;
    }
    cfg.detection.target_classes =
        yaml.getVector<int>("detection.target_classes");

    cfg.calibration_file =
        yaml.get("calibration_file", cfg.calibration_file);
    cfg.calibration_file = resolve_path(cfg.calibration_file);

    cfg.object.width_m =
        yaml.get("object.width_m", cfg.object.width_m);
    cfg.object.height_m =
        yaml.get("object.height_m", cfg.object.height_m);

    cfg.pnp.axis_length_m =
        yaml.get("pnp.axis_length_m", cfg.pnp.axis_length_m);
    cfg.pnp.yaw_sign =
        yaml.get("pnp.yaw_sign", cfg.pnp.yaw_sign);
    cfg.pnp.pitch_sign =
        yaml.get("pnp.pitch_sign", cfg.pnp.pitch_sign);
    cfg.pnp.min_z_m =
        yaml.get("pnp.min_z_m", cfg.pnp.min_z_m);
    cfg.pnp.max_range_m =
        yaml.getOptional<double>("pnp.max_range_m");

    cfg.tracking.cell_px =
        yaml.get("tracking.cell_px", cfg.tracking.cell_px);
    cfg.tracking.radius_px =
        yaml.get("tracking.radius_px", cfg.tracking.radius_px);

    cfg.calibration.pattern =
        yaml.get("calibration.pattern", cfg.calibration.pattern);
    cfg.calibration.square_m =
        yaml.get("calibration.square_m", cfg.calibration.square_m);
    cfg.calibration.frames =
        yaml.get("calibration.frames", cfg.calibration.frames);
    cfg.calibration.output =
        yaml.get("calibration.output", cfg.calibration.output);
    cfg.calibration.output = resolve_path(cfg.calibration.output);

    cfg.window.show =
        yaml.get("window.show", cfg.window.show);
    cfg.window.show_fps =
        yaml.get("window.show_fps", cfg.window.show_fps);
    cfg.window.show_rpy =
        yaml.get("window.show_rpy", cfg.window.show_rpy);
    cfg.window.show_target_panel =
        yaml.get(
            "window.show_target_panel",
            cfg.window.show_target_panel);
    cfg.window.fps_limit =
        yaml.get("window.fps_limit", cfg.window.fps_limit);

    require_positive(cfg.detection.input_size, "detection.input_size");
    require_nonnegative(cfg.detection.confidence, "detection.confidence");
    require_nonnegative(cfg.detection.nms, "detection.nms");
    require_positive(
        static_cast<double>(cfg.camera.read_max_failures),
        "camera.read_max_failures");
    require_nonnegative(
        static_cast<double>(cfg.camera.read_retry_delay_ms),
        "camera.read_retry_delay_ms");
    require_positive(cfg.exposure.min_us, "exposure.min_us");
    require_positive(cfg.exposure.max_us, "exposure.max_us");
    require_positive(cfg.exposure.max_gain, "exposure.max_gain");
    if (cfg.exposure.max_us <= cfg.exposure.min_us) {
        throw std::invalid_argument(
            "exposure.max_us must be greater than exposure.min_us");
    }
    require_nonnegative(cfg.exposure.gain_step_coarse, "exposure.gain_step_coarse");
    require_nonnegative(cfg.exposure.gain_step_fine, "exposure.gain_step_fine");
    require_positive(
        cfg.exposure.exposure_step_coarse,
        "exposure.exposure_step_coarse");
    require_positive(
        cfg.exposure.exposure_step_fine,
        "exposure.exposure_step_fine");
    require_positive(cfg.object.width_m, "object.width_m");
    require_positive(cfg.object.height_m, "object.height_m");
    require_nonnegative(cfg.pnp.min_z_m, "pnp.min_z_m");
    require_positive(
        static_cast<double>(cfg.tracking.cell_px),
        "tracking.cell_px");
    require_positive(cfg.tracking.radius_px, "tracking.radius_px");
    require_nonnegative(cfg.window.fps_limit, "window.fps_limit");
    require_positive(cfg.calibration.square_m, "calibration.square_m");
    require_positive(cfg.calibration.frames, "calibration.frames");
    if (cfg.calibration.frames < 1) {
        throw std::invalid_argument("calibration.frames must be at least 1");
    }

    return cfg;
}

}  // namespace pnp_vision::config
