#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <optional>
#include <string>

#include "pnp_vision/camera/hik.hpp"

namespace pnp_vision::camera {

class Capture {
public:
    Capture() = default;
    explicit Capture(int source);
    explicit Capture(const std::string& source);

#ifdef PNP_HAS_HIKVISION
    explicit Capture(HikCamera hik);
#endif

    bool isOpened() const;
    bool read(cv::Mat& frame);
    void release();

    bool getExposure(double& exposure) const;
    bool setExposure(double exposure);
    bool getExposureRange(
        double& min_exposure,
        double& max_exposure) const;
    bool getGain(double& gain) const;
    bool setGain(double gain);
    bool getGainRange(double& min_gain, double& max_gain) const;

private:
    std::optional<cv::VideoCapture> opencv_;
#ifdef PNP_HAS_HIKVISION
    std::optional<HikCamera> hik_;
#endif
};

Capture create_capture(
    const std::string& camera_mode,
    const std::string& source);

}  // namespace pnp_vision::camera
