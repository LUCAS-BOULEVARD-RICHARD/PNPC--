#include "pnp_vision/camera/hik.hpp"

#ifdef PNP_HAS_HIKVISION

#include <MvCameraControl.h>

#include <opencv2/imgproc.hpp>

#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

void throw_sdk_error(const char* message, int ret) {
    std::ostringstream text;
    text << message << ": 0x" << std::hex << ret;
    throw std::runtime_error(text.str());
}

}  // namespace

namespace pnp_vision::camera {

HikCamera::HikCamera() {
    openFirstDevice();
}

HikCamera::~HikCamera() {
    release();
}

HikCamera::HikCamera(HikCamera&& other) noexcept
    : handle_(other.handle_), buffer_(std::move(other.buffer_)) {
    other.handle_ = nullptr;
}

HikCamera& HikCamera::operator=(HikCamera&& other) noexcept {
    if (this != &other) {
        release();
        handle_ = other.handle_;
        other.handle_ = nullptr;
        buffer_ = std::move(other.buffer_);
    }
    return *this;
}

void HikCamera::openFirstDevice() {
    MV_CC_DEVICE_INFO_LIST device_list;
    std::memset(&device_list, 0, sizeof(device_list));

    int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
    if (ret != MV_OK) {
        throw_sdk_error("EnumDevices failed", ret);
    }
    if (device_list.nDeviceNum == 0) {
        throw std::runtime_error("no Hikvision camera found");
    }

    ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[0]);
    if (ret != MV_OK) {
        throw_sdk_error("CreateHandle failed", ret);
    }

    ret = MV_CC_OpenDevice(handle_, MV_ACCESS_Exclusive, 0);
    if (ret != MV_OK) {
        throw_sdk_error("OpenDevice failed", ret);
    }

    ret = MV_CC_SetEnumValue(handle_, "TriggerMode", 0);
    if (ret != MV_OK) {
        throw_sdk_error("Set TriggerMode failed", ret);
    }

    ret = MV_CC_SetEnumValue(handle_, "PixelFormat", PixelType_Gvsp_RGB8_Packed);
    if (ret != MV_OK) {
        throw_sdk_error("Set PixelFormat failed", ret);
    }

    ret = MV_CC_StartGrabbing(handle_);
    if (ret != MV_OK) {
        throw_sdk_error("StartGrabbing failed", ret);
    }

    MVCC_INTVALUE_EX payload;
    std::memset(&payload, 0, sizeof(payload));
    ret = MV_CC_GetIntValueEx(handle_, "PayloadSize", &payload);
    if (ret != MV_OK) {
        throw_sdk_error("Get PayloadSize failed", ret);
    }
    if (payload.nCurValue <= 0) {
        throw std::runtime_error("invalid payload size");
    }
    buffer_.assign(static_cast<size_t>(payload.nCurValue), 0);
}

bool HikCamera::isOpened() const {
    return handle_ != nullptr;
}

bool HikCamera::read(cv::Mat& frame) {
    if (!isOpened()) {
        return false;
    }

    MV_FRAME_OUT_INFO_EX frame_info;
    std::memset(&frame_info, 0, sizeof(frame_info));
    const int ret = MV_CC_GetOneFrameTimeout(
        handle_,
        buffer_.data(),
        static_cast<unsigned int>(buffer_.size()),
        &frame_info,
        1000);
    if (ret != MV_OK) {
        return false;
    }

    const int width = static_cast<int>(frame_info.nWidth);
    const int height = static_cast<int>(frame_info.nHeight);
    if (width <= 0 || height <= 0) {
        return false;
    }

    switch (frame_info.enPixelType) {
        case PixelType_Gvsp_RGB8_Packed:
            cv::cvtColor(
                cv::Mat(height, width, CV_8UC3, buffer_.data()),
                frame,
                cv::COLOR_RGB2BGR);
            break;
        case PixelType_Gvsp_BGR8_Packed:
            cv::Mat(height, width, CV_8UC3, buffer_.data()).copyTo(frame);
            break;
        case PixelType_Gvsp_Mono8:
            cv::cvtColor(
                cv::Mat(height, width, CV_8UC1, buffer_.data()),
                frame,
                cv::COLOR_GRAY2BGR);
            break;
        case PixelType_Gvsp_BayerRG8:
            cv::cvtColor(
                cv::Mat(height, width, CV_8UC1, buffer_.data()),
                frame,
                cv::COLOR_BayerRG2BGR);
            break;
        case PixelType_Gvsp_BayerGB8:
            cv::cvtColor(
                cv::Mat(height, width, CV_8UC1, buffer_.data()),
                frame,
                cv::COLOR_BayerGB2BGR);
            break;
        case PixelType_Gvsp_YUV422_YUYV_Packed:
            cv::cvtColor(
                cv::Mat(height, width, CV_8UC2, buffer_.data()),
                frame,
                cv::COLOR_YUV2BGR_YUYV);
            break;
        default: {
            std::ostringstream text;
            text << "unsupported pixel type: " << frame_info.enPixelType;
            throw std::runtime_error(text.str());
        }
    }
    return true;
}

bool HikCamera::getExposure(double& exposure_us) const {
    if (!isOpened()) {
        return false;
    }

    MVCC_FLOATVALUE value;
    std::memset(&value, 0, sizeof(value));
    if (MV_CC_GetFloatValue(handle_, "ExposureTime", &value) != MV_OK) {
        return false;
    }
    exposure_us = value.fCurValue;
    return true;
}

bool HikCamera::setExposure(double exposure_us) {
    if (!isOpened()) {
        return false;
    }

    (void)MV_CC_SetEnumValue(handle_, "ExposureAuto", 0);
    return MV_CC_SetFloatValue(
               handle_,
               "ExposureTime",
               static_cast<float>(exposure_us)) == MV_OK;
}

bool HikCamera::getExposureRange(
    double& min_exposure_us,
    double& max_exposure_us) const {
    if (!isOpened()) {
        return false;
    }

    MVCC_FLOATVALUE value;
    std::memset(&value, 0, sizeof(value));
    if (MV_CC_GetFloatValue(handle_, "ExposureTime", &value) != MV_OK) {
        return false;
    }
    min_exposure_us = value.fMin;
    max_exposure_us = value.fMax;
    return true;
}

bool HikCamera::getGain(double& gain) const {
    if (!isOpened()) {
        return false;
    }

    MVCC_FLOATVALUE value;
    std::memset(&value, 0, sizeof(value));
    if (MV_CC_GetFloatValue(handle_, "Gain", &value) != MV_OK) {
        return false;
    }
    gain = value.fCurValue;
    return true;
}

bool HikCamera::setGain(double gain) {
    if (!isOpened()) {
        return false;
    }

    (void)MV_CC_SetEnumValue(handle_, "GainAuto", 0);
    return MV_CC_SetFloatValue(
               handle_,
               "Gain",
               static_cast<float>(gain)) == MV_OK;
}

bool HikCamera::getGainRange(
    double& min_gain,
    double& max_gain) const {
    if (!isOpened()) {
        return false;
    }

    MVCC_FLOATVALUE value;
    std::memset(&value, 0, sizeof(value));
    if (MV_CC_GetFloatValue(handle_, "Gain", &value) != MV_OK) {
        return false;
    }
    min_gain = value.fMin;
    max_gain = value.fMax;
    return true;
}

void HikCamera::release() {
    if (handle_ == nullptr) {
        return;
    }
    try {
        (void)MV_CC_StopGrabbing(handle_);
        (void)MV_CC_CloseDevice(handle_);
        (void)MV_CC_DestroyHandle(handle_);
    } catch (...) {
    }
    handle_ = nullptr;
    buffer_.clear();
}

}  // namespace pnp_vision::camera

#endif
