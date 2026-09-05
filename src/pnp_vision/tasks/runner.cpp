#include "pnp_vision/tasks/runner.hpp"
#include "pnp_vision/tasks/exposure.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include "pnp_vision/camera/capture.hpp"
#include "pnp_vision/detection/detector.hpp"
#include "pnp_vision/geometry/calibration.hpp"
#include "pnp_vision/geometry/gimbal.hpp"
#include "pnp_vision/geometry/pnp.hpp"
#include "pnp_vision/visualization/drawing.hpp"
#include "pnp_vision/visualization/hud.hpp"
#include "pnp_vision/visualization/tracking.hpp"

namespace pnp_vision::tasks {

std::string format_pose(const geometry::PnPPose& pose);
std::pair<int, int> parse_pattern(const std::string& pattern);
std::vector<cv::Point3f> make_object_points(
    int pattern_width,
    int pattern_height,
    double square);

namespace {

constexpr int kKeyLeft = 0xff51;
constexpr int kKeyUp = 0xff52;
constexpr int kKeyRight = 0xff53;
constexpr int kKeyDown = 0xff54;

constexpr size_t kFrameQueueCapacity = 2;

// 有界小缓冲区。队列满时丢弃最旧的待处理帧，
// 保证处理线程始终拿到尽可能新的输入。
class LatestFrameQueue {
public:
    void push(cv::Mat frame) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_) {
                return;
            }
            if (frames_.size() >= kFrameQueueCapacity) {
                frames_.pop_front();
                ++dropped_frames_;
            }
            frames_.push_back(std::move(frame));
        }
        condition_.notify_one();
    }

    bool pop(cv::Mat& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] {
            return stopped_ || !frames_.empty();
        });
        if (frames_.empty()) {
            return false;
        }
        frame = std::move(frames_.front());
        frames_.pop_front();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
            frames_.clear();
        }
        condition_.notify_all();
    }

    size_t dropped_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_frames_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<cv::Mat> frames_;
    bool stopped_ = false;
    size_t dropped_frames_ = 0;
};

struct SharedExposure {
    std::mutex mutex;
    ExposureControl control;
    bool enabled = false;
    bool available = false;
    bool command_pending = false;
    bool command_brighter = false;
    bool command_coarse = false;
    std::string feedback;
};

void update_capture_exposure(camera::Capture& cap, SharedExposure& state) {
    ExposureControl control;
    bool should_adjust = false;
    bool brighter = false;
    bool coarse = false;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.enabled) {
            return;
        }
        control = state.control;
        should_adjust = state.command_pending;
        brighter = state.command_brighter;
        coarse = state.command_coarse;
    }

    const bool available = read_exposure_state(cap, control);
    std::string feedback;
    if (should_adjust) {
        if (adjust_brightness(cap, control, brighter, coarse)) {
            std::ostringstream text;
            text << "Gain " << std::fixed << std::setprecision(1)
                 << control.gain << "  Exposure "
                 << std::setprecision(0) << control.exposure_us << "us";
            feedback = text.str();
        } else {
            feedback = "Reached gain/exposure limit";
        }
    }

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.control = control;
        state.available = available;
        state.command_pending = false;
        if (!feedback.empty()) {
            state.feedback = std::move(feedback);
        }
    }
}

void request_exposure_adjustment(
    SharedExposure& state,
    bool brighter,
    bool coarse) {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.command_pending = true;
    state.command_brighter = brighter;
    state.command_coarse = coarse;
}

std::string take_exposure_feedback(SharedExposure& state) {
    std::lock_guard<std::mutex> lock(state.mutex);
    std::string feedback = std::move(state.feedback);
    state.feedback.clear();
    return feedback;
}

bool snapshot_exposure(SharedExposure& state, ExposureControl& control) {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.enabled) {
        return false;
    }
    control = state.control;
    return state.available;
}

struct CaptureThreadGuard {
    std::atomic<bool>& stop_requested;
    LatestFrameQueue& frames;
    std::thread& capture_thread;

    ~CaptureThreadGuard() {
        stop_requested.store(true);
        frames.stop();
        if (capture_thread.joinable()) {
            capture_thread.join();
        }
    }
};

int run_live(
    camera::Capture& cap,
    const std::string& window_name,
    bool no_show,
    double fps_limit,
    bool show_fps,
    bool draw_fps,
    const std::function<void(cv::Mat&, double)>& process,
    ExposureControl exposure_state = ExposureControl{},
    bool exposure_keys = false,
    int max_read_failures = 30,
    int read_retry_delay_ms = 50) {
    LatestFrameQueue frames;
    std::atomic<bool> stop_capture{false};
    SharedExposure exposure;
    exposure.enabled = exposure_keys;
    exposure.control = exposure_state;

    std::thread capture_thread([&] {
        int fail_count = 0;
        try {
            while (!stop_capture.load(std::memory_order_relaxed)) {
                cv::Mat frame;
                if (!cap.read(frame)) {
                    ++fail_count;
                    if (fail_count >= max_read_failures) {
                        stop_capture.store(true);
                        frames.stop();
                        break;
                    }
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(read_retry_delay_ms));
                    continue;
                }
                fail_count = 0;
                if (exposure_keys) {
                    update_capture_exposure(cap, exposure);
                }
                frames.push(std::move(frame));
            }
        } catch (const std::exception& error) {
            std::cerr << "Capture error: " << error.what() << "\n";
            stop_capture.store(true);
            frames.stop();
        }
        cap.release();
    });

    double fps = 0.0;
    bool has_last = false;
    std::chrono::steady_clock::time_point last_completed;
    int fps_console_count = 0;

    {
        CaptureThreadGuard guard{stop_capture, frames, capture_thread};
        while (true) {
            cv::Mat frame;
            if (!frames.pop(frame)) {
                break;
            }

            const std::string exposure_feedback =
                take_exposure_feedback(exposure);
            if (!exposure_feedback.empty()) {
                std::cout << exposure_feedback << "\n";
            }

            const auto frame_start = std::chrono::steady_clock::now();
            if (has_last) {
                const double dt =
                    std::chrono::duration<double>(frame_start - last_completed)
                        .count();
                if (dt > 0.0) {
                    const double instant_fps = 1.0 / dt;
                    fps = fps <= 0.0
                              ? instant_fps
                              : 0.9 * fps + 0.1 * instant_fps;
                }
            }
            has_last = true;

            process(frame, fps);
            if (draw_fps) {
                visualization::draw_fps_overlay(frame, fps, fps_limit);
            }
            if (show_fps && ++fps_console_count >= 30) {
                fps_console_count = 0;
                std::cout << "FPS " << std::fixed << std::setprecision(1)
                          << fps;
                if (fps_limit > 0.0) {
                    std::cout << " (max " << std::setprecision(0) << fps_limit
                              << ")";
                }
                const size_t dropped = frames.dropped_count();
                if (dropped > 0) {
                    std::cout << " dropped " << dropped;
                }
                std::cout << "\n";
            }

            if (fps_limit > 0.0) {
                const double target_interval = 1.0 / fps_limit;
                const double elapsed =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - frame_start)
                        .count();
                if (elapsed < target_interval) {
                    std::this_thread::sleep_for(
                        std::chrono::duration<double>(
                            target_interval - elapsed));
                }
            }
            last_completed = std::chrono::steady_clock::now();

            if (no_show) {
                continue;
            }

            ExposureControl current_exposure;
            const bool exposure_available =
                snapshot_exposure(exposure, current_exposure);
            if (exposure_available) {
                std::ostringstream text;
                text << "Gain " << std::fixed << std::setprecision(1)
                     << current_exposure.gain << "  Exposure "
                     << std::setprecision(0) << current_exposure.exposure_us
                     << "us";
                cv::putText(
                    frame,
                    text.str(),
                    cv::Point(10, frame.rows - 12),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 0),
                    1);
            }

            cv::imshow(window_name, frame);
            const int key = cv::waitKeyEx(1) & 0xFFFF;
            if (key == 'q' || key == 27) {
                break;
            }

            const bool raise_key =
                key == kKeyRight || key == kKeyUp ||
                key == '=' || key == '+';
            const bool lower_key =
                key == kKeyLeft || key == kKeyDown ||
                key == '-' || key == '_';
            if (!exposure_keys || !(raise_key || lower_key)) {
                continue;
            }

            const bool coarse =
                key == kKeyRight || key == kKeyUp ||
                key == kKeyLeft || key == kKeyDown;
            request_exposure_adjustment(exposure, raise_key, coarse);
        }
    }

    cv::destroyAllWindows();
    return 0;
}

int run_calibration(const TaskArgs& args) {
    const auto [pattern_width, pattern_height] = parse_pattern(args.pattern);
    const cv::Size pattern(pattern_width, pattern_height);
    camera::Capture cap = camera::create_capture(args.camera, args.source);
    if (!cap.isOpened()) {
        throw std::runtime_error("cannot open source: " + args.source);
    }

    const std::vector<cv::Point3f> objp =
        make_object_points(pattern_width, pattern_height, args.square);
    std::vector<std::vector<cv::Point3f>> obj_points;
    std::vector<std::vector<cv::Point2f>> img_points;
    cv::Size image_size;
    int captured = 0;
    int fail_count = 0;

    std::cout << "Target: " << args.count
              << " frames. Press c to capture, q to finish, Esc to abort.\n";
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            ++fail_count;
            if (fail_count >= args.camera_max_failures) {
                break;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(args.camera_retry_delay_ms));
            continue;
        }
        fail_count = 0;
        if (image_size.empty()) {
            image_size = frame.size();
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        const bool found = cv::findChessboardCorners(gray, pattern, corners);
        if (found) {
            const cv::TermCriteria criteria(
                cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                30,
                0.001);
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            cv::drawChessboardCorners(frame, pattern, corners, found);
        }

        std::ostringstream text;
        text << "captured " << captured << "/" << args.count;
        cv::putText(
            frame,
            text.str(),
            cv::Point(10, 28),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            2);
        cv::imshow("Calibration", frame);
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'c' && found) {
            obj_points.push_back(objp);
            img_points.push_back(corners);
            ++captured;
            std::cout << "Captured " << captured << "/" << args.count << "\n";
        } else if (key == 'q' || key == 27) {
            break;
        }
        if (captured >= args.count) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    if (obj_points.size() < 5) {
        throw std::runtime_error("not enough calibration frames captured");
    }

    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    const double rms = cv::calibrateCamera(
        obj_points,
        img_points,
        image_size,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs);

    cv::FileStorage fs(args.output, cv::FileStorage::WRITE);
    fs.write("camera_matrix", camera_matrix);
    fs.write("dist_coeffs", dist_coeffs);
    fs.write("image_width", image_size.width);
    fs.write("image_height", image_size.height);
    fs.release();

    std::cout << "RMS error: " << std::fixed << std::setprecision(4) << rms << "\n";
    std::cout << "Saved calibration to " << args.output << "\n";
    return 0;
}

int run_detection_task(
    const TaskArgs& args,
    detection::YoloDetector& detector,
    camera::Capture& cap) {
    return run_live(
        cap,
        "YOLO Task1",
        args.no_show,
        args.fps_limit,
        args.show_fps,
        true,
        [&](cv::Mat& frame, double) {
            const std::vector<detection::Detection> detections =
                detector.detectBgr(frame, args.conf, args.classes);
            for (const detection::Detection& det : detections) {
                std::cout << "DET cls=" << det.cls << " " << det.name
                          << " conf=" << std::fixed << std::setprecision(2) << det.conf
                          << " xyxy=" << det.xyxy[0] << "," << det.xyxy[1] << ","
                          << det.xyxy[2] << "," << det.xyxy[3] << "\n";
                visualization::draw_detection(frame, det);
            }
        },
        ExposureControl{},
        false,
        args.camera_max_failures,
        args.camera_retry_delay_ms);
}

int run_distance_task(
    const TaskArgs& args,
    detection::YoloDetector& detector,
    camera::Capture& cap,
    const cv::Mat& obj_points,
    const geometry::CalibrationData& calib) {
    return run_live(
        cap,
        "YOLO + PnP Task2",
        args.no_show,
        args.fps_limit,
        args.show_fps,
        true,
        [&](cv::Mat& frame, double) {
            const std::vector<detection::Detection> detections =
                detector.detectBgr(frame, args.conf, args.classes);
            for (const detection::Detection& det : detections) {
                const std::optional<std::array<double, 4>> pose =
                    geometry::solve_pnp(
                        det,
                        obj_points,
                        calib.camera_matrix,
                        calib.dist_coeffs);
                if (!pose.has_value()) {
                    continue;
                }
                const auto& [x, y, z, dist] = *pose;
                std::cout << "PnP cls=" << det.cls << " " << det.name
                          << " conf=" << std::fixed << std::setprecision(2) << det.conf
                          << " xyxy=" << det.xyxy[0] << "," << det.xyxy[1] << ","
                          << det.xyxy[2] << "," << det.xyxy[3]
                          << " X=" << std::setprecision(3) << x
                          << " Y=" << y << " Z=" << z << " dist=" << dist << "\n";
                std::ostringstream extra;
                extra << "Z=" << std::fixed << std::setprecision(2) << z << "m";
                visualization::draw_detection(frame, det, extra.str());
            }
        },
        ExposureControl{},
        false,
        args.camera_max_failures,
        args.camera_retry_delay_ms);
}

int run_pose_task(
    const TaskArgs& args,
    detection::YoloDetector& detector,
    camera::Capture& cap,
    const cv::Mat& obj_points,
    const geometry::CalibrationData& calib) {
    return run_live(
        cap,
        "YOLO + PnP Task3",
        args.no_show,
        args.fps_limit,
        args.show_fps,
        true,
        [&](cv::Mat& frame, double) {
            const std::vector<detection::Detection> detections =
                detector.detectBgr(frame, args.conf, args.classes);
            for (const detection::Detection& det : detections) {
                const std::optional<geometry::PnPPose> pose =
                    geometry::solve_pnp_pose(
                        det,
                        obj_points,
                        calib.camera_matrix,
                        calib.dist_coeffs);
                if (!pose.has_value()) {
                    visualization::draw_detection(frame, det, "PnP failed");
                    continue;
                }

                const cv::Vec3d euler =
                    geometry::rvec_to_euler_zyx(pose->rvec());
                std::cout << "POSE cls=" << det.cls << " " << det.name
                          << " conf=" << std::fixed << std::setprecision(2) << det.conf
                          << " " << format_pose(*pose) << "\n";
                std::ostringstream extra;
                extra << "R=" << std::fixed << std::setprecision(1) << euler[0]
                      << " P=" << euler[1] << " Y=" << euler[2]
                      << " Z=" << std::setprecision(2) << pose->z() << "m";
                visualization::draw_detection(frame, det, extra.str());
                visualization::draw_pnp_axes(
                    frame,
                    calib.camera_matrix,
                    calib.dist_coeffs,
                    *pose,
                    args.axis_length);
            }
        },
        ExposureControl{},
        false,
        args.camera_max_failures,
        args.camera_retry_delay_ms);
}

int run_visualize_task(
    const TaskArgs& args,
    detection::YoloDetector& detector,
    camera::Capture& cap,
    const cv::Mat& obj_points,
    const geometry::CalibrationData& calib) {
    visualization::PoseMap last_pose_by_key;

    return run_live(
        cap,
        "YOLO + PnP Task4",
        args.no_show,
        args.fps_limit,
        args.show_fps,
        false,
        [&](cv::Mat& frame, double fps) {
            std::vector<detection::Detection> detections =
                detector.detectBgr(frame, args.conf, args.classes);
            // 单目标流程：只保留置信度最高的检测结果，
            // HUD/侧边栏始终只描述一个瞄准目标。
            detections = detection::keep_highest_confidence(std::move(detections));

            visualization::TargetSolveResult solved =
                visualization::solve_targets(
                    detections,
                    obj_points,
                    calib.camera_matrix,
                    calib.dist_coeffs,
                    last_pose_by_key,
                    args.tracking.cell_px,
                    args.tracking.radius_px);

            for (visualization::TargetOverlay& target : solved.targets) {
                if (!target.pose.has_value()) {
                    continue;
                }
                try {
                    target.gimbal_rpy = geometry::calc_gimbal_target_rpy(
                        target.pose->tvec(),
                        args.yaw_sign,
                        args.pitch_sign,
                        args.min_z,
                        args.max_range);
                } catch (const geometry::GimbalError&) {
                    target.gimbal_rpy.reset();
                    if (target.status == visualization::TargetStatus::Ok) {
                        --solved.ok_count;
                    } else if (target.status == visualization::TargetStatus::Stale) {
                        --solved.stale_count;
                    }
                    ++solved.fail_count;
                    target.status = visualization::TargetStatus::Invalid;
                    target.color = visualization::error_color();
                }
            }

            for (const visualization::TargetOverlay& target : solved.targets) {
                visualization::draw_box(frame, target.det, target.color);
                if (target.pose.has_value()) {
                    visualization::draw_pnp_axes(
                        frame,
                        calib.camera_matrix,
                        calib.dist_coeffs,
                        *target.pose,
                        args.axis_length);
                }
            }

            visualization::draw_top_left_hud(
                frame,
                solved.targets,
                solved.ok_count,
                solved.stale_count,
                solved.fail_count,
                fps,
                args.show_fps,
                args.show_rpy);
            if (args.show_target_panel) {
                visualization::draw_target_panel(
                    frame,
                    solved.targets,
                    solved.ok_count,
                    solved.stale_count,
                    solved.fail_count);
            }

            for (size_t index = 0; index < solved.targets.size(); ++index) {
                const visualization::TargetOverlay& target = solved.targets[index];
                std::cout << "TARGET #" << (index + 1) << " " << target.det.name
                          << " conf=" << std::fixed << std::setprecision(2)
                          << target.det.conf
                          << " status=" << visualization::status_text(target.status);
                if (target.gimbal_rpy.has_value()) {
                    std::cout << " X=" << std::setprecision(3) << target.pose->x()
                              << " Y=" << target.pose->y()
                              << " Z=" << target.pose->z()
                              << " D=" << target.pose->distance()
                              << " GimbalRoll=" << std::setprecision(2)
                              << target.gimbal_rpy->val[0] * 180.0 / CV_PI
                              << " GimbalPitch=" << target.gimbal_rpy->val[1] * 180.0 / CV_PI
                              << " GimbalYaw=" << target.gimbal_rpy->val[2] * 180.0 / CV_PI;
                } else if (target.pose.has_value()) {
                    std::cout << " X=" << std::setprecision(3) << target.pose->x()
                              << " Y=" << target.pose->y()
                              << " Z=" << target.pose->z()
                              << " D=" << target.pose->distance() << " gimbal=None";
                } else {
                    std::cout << " pose=None";
                }
                std::cout << "\n";
            }
        },
        make_exposure_control(args.exposure),
        true,
        args.camera_max_failures,
        args.camera_retry_delay_ms);
}

int run_task(const TaskArgs& args) {
    const bool needs_pnp =
        args.kind == TaskKind::Distance ||
        args.kind == TaskKind::Pose ||
        args.kind == TaskKind::Visualize;

    geometry::CalibrationData calib;
    cv::Mat obj_points;
    if (needs_pnp) {
        calib = geometry::load_calibration(args.calib);
        obj_points = geometry::object_points(args.object_width, args.object_height);
    }

    detection::YoloDetector detector(
        args.model,
        args.class_names,
        args.conf,
        args.nms,
        args.imgsz,
        args.runtime);
    camera::Capture cap = camera::create_capture(args.camera, args.source);
    if (!cap.isOpened()) {
        throw std::runtime_error("cannot open source: " + args.source);
    }

    std::cout << "Running. Press q or Esc to quit.\n";
    if (args.kind == TaskKind::Visualize) {
        std::cout << "Gain first, exposure "
                  << std::fixed << std::setprecision(0)
                  << args.exposure.min_us << "-" << args.exposure.max_us
                  << "us. arrows coarse, =/- fine.\n";
    }
    if (args.kind == TaskKind::Pose || args.kind == TaskKind::Visualize) {
        std::cout << "Angles are degrees, ZYX order: Rz(yaw) @ Ry(pitch) @ Rx(roll).\n";
    }

    switch (args.kind) {
        case TaskKind::Detection:
            return run_detection_task(args, detector, cap);
        case TaskKind::Distance:
            return run_distance_task(args, detector, cap, obj_points, calib);
        case TaskKind::Pose:
            return run_pose_task(args, detector, cap, obj_points, calib);
        case TaskKind::Visualize:
            return run_visualize_task(args, detector, cap, obj_points, calib);
        case TaskKind::Calibration:
            break;
    }
    throw std::runtime_error("unsupported task kind");
}

}  // namespace

int main_task(int argc, char** argv, TaskKind kind) {
    try {
        const TaskArgs args = parse_task_args(argc, argv, kind);
        if (kind == TaskKind::Calibration) {
            return run_calibration(args);
        }
        if (kind == TaskKind::Exposure) {
            return run_exposure(args);
        }
        return run_task(args);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}

}  // namespace pnp_vision::tasks
