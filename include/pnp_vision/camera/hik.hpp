#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#ifdef PNP_HAS_HIKVISION

namespace pnp_vision::camera {

class HikCamera {
public:
    HikCamera();
    ~HikCamera();

    HikCamera(HikCamera&& other) noexcept;
    HikCamera& operator=(HikCamera&& other) noexcept;

    HikCamera(const HikCamera&) = delete;
    HikCamera& operator=(const HikCamera&) = delete;

    bool isOpened() const;
    bool read(cv::Mat& frame);
    void release();

    bool getExposure(double& exposure_us) const;
    bool setExposure(double exposure_us);
    bool getExposureRange(
        double& min_exposure_us,
        double& max_exposure_us) const;
    bool getGain(double& gain) const;
    bool setGain(double gain);
    bool getGainRange(double& min_gain, double& max_gain) const;

private:
    void openFirstDevice();

    void* handle_ = nullptr;
    std::vector<unsigned char> buffer_;
};

}  // namespace pnp_vision::camera

#endif
