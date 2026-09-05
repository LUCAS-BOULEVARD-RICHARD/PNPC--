#include "pnp_vision/tasks/cli.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace pnp_vision::tasks {

namespace {

std::string kind_name(TaskKind kind) {
    switch (kind) {
        case TaskKind::Detection:
            return "Detection";
        case TaskKind::Distance:
            return "Distance";
        case TaskKind::Pose:
            return "Pose";
        case TaskKind::Visualize:
            return "Visualize";
        case TaskKind::Calibration:
            return "Calibration";
        case TaskKind::Exposure:
            return "Exposure";
    }
    return "Unknown";
}

std::string require_value(int argc, char** argv, int* index, const std::string& option) {
    if (*index + 1 >= argc) {
        throw std::runtime_error("Missing value for " + option);
    }
    ++*index;
    return argv[*index];
}

double parse_double(const std::string& value, const std::string& option) {
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid numeric value for " + option + ": " + value);
    }
}

int parse_int(const std::string& value, const std::string& option) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer value for " + option + ": " + value);
    }
}

bool file_exists(const std::string& path) {
    std::ifstream stream(path);
    return stream.good();
}

std::optional<std::string> find_config_path(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config") {
            return require_value(argc, argv, &i, "--config");
        }
    }
    return std::nullopt;
}

std::optional<std::string> auto_config_path(int argc, char** argv) {
    if (file_exists(config::kDefaultConfigPath)) {
        return config::kDefaultConfigPath;
    }

    std::string executable = argc > 0 && argv[0] != nullptr ? argv[0] : "";
    const size_t slash = executable.find_last_of('/');
    if (slash == std::string::npos || slash == 0) {
        return std::nullopt;
    }

    const std::string sibling =
        executable.substr(0, slash) + "/../config/config.yaml";
    if (file_exists(sibling)) {
        return sibling;
    }
    return std::nullopt;
}

void apply_config(const config::AppConfig& cfg, TaskArgs& args) {
    args.camera = cfg.camera.mode;
    args.source = cfg.camera.source;
    args.model = cfg.detection.model_path;
    args.runtime = cfg.detection.runtime;
    args.calib = cfg.calibration_file;
    args.exposure = cfg.exposure;
    args.tracking = cfg.tracking;
    args.camera_max_failures = cfg.camera.read_max_failures;
    args.camera_retry_delay_ms = cfg.camera.read_retry_delay_ms;
    args.object_width = cfg.object.width_m;
    args.object_height = cfg.object.height_m;
    args.axis_length = cfg.pnp.axis_length_m;
    args.conf = cfg.detection.confidence;
    args.nms = cfg.detection.nms;
    args.imgsz = cfg.detection.input_size;
    args.classes = cfg.detection.target_classes;
    args.class_names = cfg.detection.class_names;
    args.no_show = !cfg.window.show;
    args.show_fps = cfg.window.show_fps;
    args.show_rpy = cfg.window.show_rpy;
    args.show_target_panel = cfg.window.show_target_panel;
    args.fps_limit = cfg.window.fps_limit;
    args.yaw_sign = cfg.pnp.yaw_sign;
    args.pitch_sign = cfg.pnp.pitch_sign;
    args.min_z = cfg.pnp.min_z_m;
    args.max_range = cfg.pnp.max_range_m;
    args.pattern = cfg.calibration.pattern;
    args.square = cfg.calibration.square_m;
    args.count = cfg.calibration.frames;
    args.output = cfg.calibration.output;
}

}  // namespace

TaskArgs::TaskArgs() {
    const config::AppConfig cfg = config::default_config();
    camera = cfg.camera.mode;
    source = cfg.camera.source;
    model = cfg.detection.model_path;
    runtime = cfg.detection.runtime;
    calib = cfg.calibration_file;
    exposure = cfg.exposure;
    tracking = cfg.tracking;
    camera_max_failures = cfg.camera.read_max_failures;
    camera_retry_delay_ms = cfg.camera.read_retry_delay_ms;
    object_width = cfg.object.width_m;
    object_height = cfg.object.height_m;
    axis_length = cfg.pnp.axis_length_m;
    conf = cfg.detection.confidence;
    nms = cfg.detection.nms;
    imgsz = cfg.detection.input_size;
    classes = cfg.detection.target_classes;
    class_names = cfg.detection.class_names;
    no_show = !cfg.window.show;
    show_fps = cfg.window.show_fps;
    show_rpy = cfg.window.show_rpy;
    show_target_panel = cfg.window.show_target_panel;
    fps_limit = cfg.window.fps_limit;
    yaw_sign = cfg.pnp.yaw_sign;
    pitch_sign = cfg.pnp.pitch_sign;
    min_z = cfg.pnp.min_z_m;
    max_range = cfg.pnp.max_range_m;
    pattern = cfg.calibration.pattern;
    square = cfg.calibration.square_m;
    count = cfg.calibration.frames;
    output = cfg.calibration.output;
}

void print_usage(TaskKind kind) {
    std::cout
        << "Usage for " << kind_name(kind) << ":\n"
        << "  --config <config.yaml> (default config.yaml when present)\n"
        << "  --camera laptop|hik\n"
        << "  --source <camera index or video path>\n"
        << "  --model <model path>\n"
        << "  --runtime auto|tensorrt|libtorch|onnx\n"
        << "  --calib <calibration.yaml>\n"
        << "  --conf <float>\n"
        << "  --nms <float>\n"
        << "  --imgsz <int>\n"
        << "  --classes <class ids...>\n"
        << "  --no-show\n"
        << "  --fps-limit <fps> (0 disables the cap; default from config)\n";
    if (kind == TaskKind::Distance || kind == TaskKind::Pose ||
        kind == TaskKind::Visualize) {
        std::cout
            << "  --object-width <meters>\n"
            << "  --object-height <meters>\n";
    }
    if (kind == TaskKind::Pose || kind == TaskKind::Visualize) {
        std::cout << "  --axis-length <meters>\n";
    }
    if (kind == TaskKind::Visualize) {
        std::cout
            << "  --yaw-sign <float>\n"
            << "  --pitch-sign <float>\n"
            << "  --min-z <meters>\n"
            << "  --max-range <meters>\n";
    }
    if (kind == TaskKind::Calibration) {
        std::cout
            << "  --pattern <WxH inner corners>\n"
            << "  --square <meters>\n"
            << "  --count <frames>\n"
            << "  --output <calibration.yaml>\n";
    }
}

TaskArgs parse_task_args(int argc, char** argv, TaskKind kind) {
    TaskArgs args;
    args.kind = kind;
    bool config_loaded = false;

    const std::optional<std::string> explicit_config = find_config_path(argc, argv);
    if (explicit_config.has_value()) {
        const config::AppConfig cfg = config::load_config(*explicit_config);
        apply_config(cfg, args);
        config_loaded = true;
    } else if (const std::optional<std::string> auto_config =
                   auto_config_path(argc, argv);
               auto_config.has_value()) {
        const config::AppConfig cfg = config::load_config(*auto_config);
        apply_config(cfg, args);
        config_loaded = true;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(kind);
            std::exit(0);
        } else if (arg == "--config") {
            require_value(argc, argv, &i, arg);
        } else if (arg == "--camera") {
            args.camera = require_value(argc, argv, &i, arg);
        } else if (arg == "--source") {
            args.source = require_value(argc, argv, &i, arg);
        } else if (arg == "--model") {
            args.model = require_value(argc, argv, &i, arg);
        } else if (arg == "--runtime") {
            args.runtime = require_value(argc, argv, &i, arg);
        } else if (arg == "--calib") {
            args.calib = require_value(argc, argv, &i, arg);
        } else if (arg == "--object-width") {
            args.object_width =
                parse_double(require_value(argc, argv, &i, arg), arg);
            args.has_object_width = true;
        } else if (arg == "--object-height") {
            args.object_height =
                parse_double(require_value(argc, argv, &i, arg), arg);
            args.has_object_height = true;
        } else if (arg == "--axis-length") {
            args.axis_length = parse_double(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--conf") {
            args.conf = parse_double(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--nms") {
            args.nms = parse_double(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--imgsz") {
            args.imgsz = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--classes") {
            ++i;
            args.classes.clear();
            while (i < argc && argv[i][0] != '-') {
                args.classes.push_back(parse_int(argv[i], arg));
                ++i;
            }
            --i;
        } else if (arg == "--no-show") {
            args.no_show = true;
        } else if (arg == "--fps-limit") {
            args.fps_limit = parse_double(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--yaw-sign") {
            args.yaw_sign = parse_double(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--pitch-sign") {
            args.pitch_sign = parse_double(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--min-z") {
            args.min_z = parse_double(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--max-range") {
            const std::string value = require_value(argc, argv, &i, arg);
            args.max_range = value == "none" ? std::nullopt
                                             : std::optional<double>(parse_double(value, arg));
        } else if (arg == "--pattern") {
            args.pattern = require_value(argc, argv, &i, arg);
        } else if (arg == "--square") {
            args.square = parse_double(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--count") {
            args.count = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--output") {
            args.output = require_value(argc, argv, &i, arg);
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (config_loaded) {
        args.has_object_width = true;
        args.has_object_height = true;
    }
    if (args.fps_limit < 0.0) {
        throw std::runtime_error("--fps-limit must not be negative");
    }

    if (kind == TaskKind::Distance || kind == TaskKind::Pose ||
        kind == TaskKind::Visualize) {
        if (!args.has_object_width || !args.has_object_height) {
            throw std::runtime_error(
                "--object-width and --object-height are required");
        }
    }
    return args;
}

}  // namespace pnp_vision::tasks
