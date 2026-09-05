#include "pnp_vision/camera/capture.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "pnp_vision/util/text.hpp"

namespace {

// Linux V4L2 的 exposure_time_absolute 以 100 微秒为单位。
constexpr double kOpenCvExposureUnitUs = 100.0;

bool is_digits(const std::string& text) {
    return !text.empty() &&
           text.find_first_not_of("0123456789") == std::string::npos;
}

}  // namespace

namespace pnp_vision::camera {

Capture::Capture(int source)
    : opencv_(cv::VideoCapture(source)) {}

Capture::Capture(const std::string& source)
    : opencv_(cv::VideoCapture(source)) {}

#ifdef PNP_HAS_HIKVISION
Capture::Capture(HikCamera hik)
    : hik_(std::move(hik)) {}
#endif

bool Capture::isOpened() const {
    if (opencv_.has_value()) {
        return opencv_->isOpened();
    }
#ifdef PNP_HAS_HIKVISION
    if (hik_.has_value()) {
        return hik_->isOpened();
    }
#endif
    return false;
}

bool Capture::read(cv::Mat& frame) {
    if (opencv_.has_value()) {
        return opencv_->read(frame);
    }
#ifdef PNP_HAS_HIKVISION
    if (hik_.has_value()) {
        return hik_->read(frame);
    }
#endif
    return false;
}

bool Capture::getExposure(double& exposure) const {
    if (opencv_.has_value()) {
        const double raw_exposure = opencv_->get(cv::CAP_PROP_EXPOSURE);
        if (raw_exposure == -1.0) {
            return false;
        }
        exposure = raw_exposure * kOpenCvExposureUnitUs;
        return true;
    }
#ifdef PNP_HAS_HIKVISION
    if (hik_.has_value()) {
        return hik_->getExposure(exposure);
    }
#endif
    return false;
}

bool Capture::setExposure(double exposure) {
    if (opencv_.has_value()) {
        if (!opencv_->set(cv::CAP_PROP_AUTO_EXPOSURE, 1.0)) {
            (void)opencv_->set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
        }
        return opencv_->set(
            cv::CAP_PROP_EXPOSURE,
            exposure / kOpenCvExposureUnitUs);
    }
#ifdef PNP_HAS_HIKVISION
    if (hik_.has_value()) {
        return hik_->setExposure(exposure);
    }
#endif
    return false;
}

bool Capture::getExposureRange(
    double& min_exposure,
    double& max_exposure) const {
    if (opencv_.has_value()) {
        return false;
    }
#ifdef PNP_HAS_HIKVISION
    if (hik_.has_value()) {
        return hik_->getExposureRange(min_exposure, max_exposure);
    }
#endif
    return false;
}

bool Capture::getGain(double& gain) const {
    if (opencv_.has_value()) {
        gain = opencv_->get(cv::CAP_PROP_GAIN);
        return gain != -1.0;
    }
#ifdef PNP_HAS_HIKVISION
    if (hik_.has_value()) {
        return hik_->getGain(gain);
    }
#endif
    return false;
}

bool Capture::setGain(double gain) {
    if (opencv_.has_value()) {
        return opencv_->set(cv::CAP_PROP_GAIN, gain);
    }
#ifdef PNP_HAS_HIKVISION
    if (hik_.has_value()) {
        return hik_->setGain(gain);
    }
#endif
    return false;
}

bool Capture::getGainRange(
    double& min_gain,
    double& max_gain) const {
    if (opencv_.has_value()) {
        return false;
    }
#ifdef PNP_HAS_HIKVISION
    if (hik_.has_value()) {
        return hik_->getGainRange(min_gain, max_gain);
    }
#endif
    return false;
}

void Capture::release() {
    if (opencv_.has_value()) {
        opencv_.reset();
    }
#ifdef PNP_HAS_HIKVISION
    if (hik_.has_value()) {
        hik_.reset();
    }
#endif
}

Capture create_capture(
    const std::string& camera_mode,
    const std::string& source) {
    const std::string mode = util::lower_ascii(camera_mode);
    const std::string lowered_source = util::lower_ascii(source);

    if (mode == "hik" || lowered_source == "hik") {
#ifdef PNP_HAS_HIKVISION
        return Capture(HikCamera());
#else
        throw std::runtime_error(
            "Hikvision support is not compiled");
#endif
    }

    if (is_digits(source)) {
        return Capture(std::stoi(source));
    }
    return Capture(source);
}

}  // namespace pnp_vision::camera
