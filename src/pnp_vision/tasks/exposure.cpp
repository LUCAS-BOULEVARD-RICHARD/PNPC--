#include "pnp_vision/tasks/exposure.hpp"

#include "pnp_vision/camera/capture.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace pnp_vision::tasks {

namespace {

constexpr int kKeyLeft = 0xff51;
constexpr int kKeyUp = 0xff52;
constexpr int kKeyRight = 0xff53;
constexpr int kKeyDown = 0xff54;

}  // namespace

ExposureControl make_exposure_control(
    const config::ExposureSettings& settings) {
    ExposureControl control;
    control.gain_max = std::max(1.0, settings.max_gain);
    control.exposure_us = settings.min_us;
    control.exposure_min_us = settings.min_us;
    control.exposure_max_us = settings.max_us;
    control.gain_step_coarse = settings.gain_step_coarse;
    control.gain_step_fine = settings.gain_step_fine;
    control.exposure_step_coarse = settings.exposure_step_coarse;
    control.exposure_step_fine = settings.exposure_step_fine;
    return control;
}

double clamp_exposure_us(
    const ExposureControl& control,
    double exposure_us) {
    return std::clamp(
        exposure_us,
        control.exposure_min_us,
        control.exposure_max_us);
}

double gain_step(const ExposureControl& control, bool coarse) {
    const double span =
        std::max(1.0, control.gain_max - control.gain_min);
    return std::max(
        1.0,
        span * (coarse ? control.gain_step_coarse : control.gain_step_fine));
}

bool read_exposure_state(camera::Capture& cap, ExposureControl& control) {
    double value = 0.0;
    if (!cap.getExposure(value) || value <= 0.0) {
        return false;
    }
    control.exposure_us = clamp_exposure_us(control, value);

    if (cap.getGain(value) && value >= 0.0) {
        control.gain = value;
    }
    double min_gain = control.gain_min;
    double max_gain = control.gain_max;
    if (cap.getGainRange(min_gain, max_gain) && max_gain > min_gain) {
        control.gain_min = min_gain;
        control.gain_max = max_gain;
    }
    control.gain =
        std::clamp(control.gain, control.gain_min, control.gain_max);
    return true;
}

bool adjust_brightness(
    camera::Capture& cap,
    ExposureControl& control,
    bool brighter,
    bool coarse) {
    if (brighter) {
        if (control.gain < control.gain_max - 1e-6) {
            const double next_gain = std::min(
                control.gain_max,
                control.gain + gain_step(control, coarse));
            if (cap.setGain(next_gain)) {
                control.gain = next_gain;
                control.exposure_us =
                    clamp_exposure_us(control, control.exposure_us);
                (void)cap.setExposure(control.exposure_us);
                return true;
            }
        }

        const double next_exposure = std::min(
            control.exposure_max_us,
            control.exposure_us *
                (coarse ? control.exposure_step_coarse
                        : control.exposure_step_fine));
        if (next_exposure <= control.exposure_us + 1e-9 ||
            !cap.setExposure(next_exposure)) {
            return false;
        }
        control.exposure_us = next_exposure;
        return true;
    }

    if (control.exposure_us > control.exposure_min_us + 1e-6) {
        const double next_exposure = std::max(
            control.exposure_min_us,
            control.exposure_us /
                (coarse ? control.exposure_step_coarse
                        : control.exposure_step_fine));
        if (!cap.setExposure(next_exposure)) {
            return false;
        }
        control.exposure_us = next_exposure;
        return true;
    }

    if (control.gain > control.gain_min + 1e-6) {
        const double next_gain = std::max(
            control.gain_min,
            control.gain - gain_step(control, coarse));
        if (!cap.setGain(next_gain)) {
            return false;
        }
        control.gain = next_gain;
        control.exposure_us =
            clamp_exposure_us(control, control.exposure_us);
        (void)cap.setExposure(control.exposure_us);
        return true;
    }
    return false;
}

int run_exposure(const TaskArgs& args) {
    camera::Capture cap = camera::create_capture(args.camera, args.source);
    if (!cap.isOpened()) {
        throw std::runtime_error("cannot open source: " + args.source);
    }

    ExposureControl exposure = make_exposure_control(args.exposure);
    if (!read_exposure_state(cap, exposure) ||
        !cap.setExposure(exposure.exposure_us)) {
        throw std::runtime_error(
            "manual exposure is not supported by this camera");
    }

    std::cout
        << "Gain first, then exposure. arrows coarse, - = fine.\n";
    int fail_count = 0;
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

        (void)read_exposure_state(cap, exposure);

        std::ostringstream text;
        text << "Gain " << std::fixed << std::setprecision(1)
             << exposure.gain << "  Exposure " << std::setprecision(0)
             << exposure.exposure_us << "us";
        cv::putText(
            frame,
            text.str(),
            cv::Point(10, 28),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            2);
        cv::putText(
            frame,
            "arrows coarse   - = fine   q quit",
            cv::Point(10, 56),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 0),
            1);
        cv::imshow("Exposure", frame);
        const int key = cv::waitKeyEx(1) & 0xFFFF;
        if (key == 'q' || key == 27) {
            break;
        }

        const bool lower_key =
            key == kKeyLeft || key == kKeyDown ||
            key == '-' || key == '_';
        const bool raise_key =
            key == kKeyRight || key == kKeyUp ||
            key == '=' || key == '+';
        if (lower_key || raise_key) {
            const bool coarse =
                key == kKeyRight || key == kKeyUp ||
                key == kKeyLeft || key == kKeyDown;
            if (adjust_brightness(cap, exposure, raise_key, coarse)) {
                std::cout << "Gain " << std::fixed << std::setprecision(1)
                          << exposure.gain << "  Exposure "
                          << std::setprecision(0) << exposure.exposure_us
                          << "us\n";
            } else {
                std::cout << "Reached gain/exposure limit\n";
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

}  // namespace pnp_vision::tasks
