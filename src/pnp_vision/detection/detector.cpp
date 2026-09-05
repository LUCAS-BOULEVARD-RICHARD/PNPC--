#include "pnp_vision/detection/detector.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef PNP_USE_TENSORRT
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#endif

#ifdef PNP_USE_LIBTORCH
#include <torch/script.h>
#endif

#include "pnp_vision/util/text.hpp"

namespace {

struct Candidate {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    float score = 0.0F;
    int cls = -1;
};

cv::Mat letterbox(
    const cv::Mat& src,
    int target_width,
    int target_height,
    int& pad_x,
    int& pad_y,
    float& scale_x,
    float& scale_y) {
    const double scale = std::min(
        static_cast<double>(target_width) / src.cols,
        static_cast<double>(target_height) / src.rows);
    const int new_width = std::max(1, static_cast<int>(std::lround(src.cols * scale)));
    const int new_height = std::max(1, static_cast<int>(std::lround(src.rows * scale)));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_width, new_height), 0.0, 0.0, cv::INTER_LINEAR);

    pad_x = (target_width - new_width) / 2;
    pad_y = (target_height - new_height) / 2;
    scale_x = static_cast<float>(src.cols) / new_width;
    scale_y = static_cast<float>(src.rows) / new_height;

    cv::Mat canvas(target_height, target_width, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_width, new_height)));
    return canvas;
}

}  // namespace

namespace pnp_vision::detection {

#ifdef PNP_USE_TENSORRT

class TensorRtInference {
public:
    explicit TensorRtInference(const std::string& engine_path);
    ~TensorRtInference();

    TensorRtInference(const TensorRtInference&) = delete;
    TensorRtInference& operator=(const TensorRtInference&) = delete;

    cv::Mat infer(const cv::Mat& letterboxed_rgb);

private:
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity, char const*) noexcept override {}
    };

    static void check_cuda(cudaError_t error, const char* message) {
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string(message) + ": " + cudaGetErrorString(error));
        }
    }

    static void require(bool ok, const std::string& message) {
        if (!ok) {
            throw std::runtime_error(message);
        }
    }

    void release() noexcept;

    Logger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    void* input_device_ = nullptr;
    void* output_device_ = nullptr;
    std::string input_name_;
    std::string output_name_;
    int64_t input_elements_ = 0;
    std::vector<float> output_host_;
    std::vector<int64_t> output_shape_;
};

TensorRtInference::TensorRtInference(const std::string& engine_path) {
    try {
        const std::string lower_path = util::lower_ascii(engine_path);
        require(
            util::ends_with(lower_path, ".engine") ||
                util::ends_with(lower_path, ".trt"),
            "TensorRT requires a .engine file: " + engine_path);

        std::ifstream file(engine_path, std::ios::binary);
        require(static_cast<bool>(file), "cannot open engine: " + engine_path);
        file.seekg(0, std::ios::end);
        const std::streamsize size = file.tellg();
        require(size > 0, "engine file is empty: " + engine_path);
        file.seekg(0, std::ios::beg);
        std::vector<char> engine_data(static_cast<size_t>(size));
        file.read(engine_data.data(), size);
        require(static_cast<bool>(file), "failed to read engine: " + engine_path);

        initLibNvInferPlugins(&logger_, "");
        runtime_ = nvinfer1::createInferRuntime(logger_);
        require(runtime_ != nullptr, "failed to create TensorRT runtime");
        engine_ = runtime_->deserializeCudaEngine(
            engine_data.data(),
            engine_data.size());
        require(engine_ != nullptr, "failed to deserialize TensorRT engine");
        context_ = engine_->createExecutionContext();
        require(context_ != nullptr, "failed to create TensorRT context");

        for (int32_t index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* name = engine_->getIOTensorName(index);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                require(input_name_.empty(), "engine has multiple inputs");
                input_name_ = name;
            } else if (
                engine_->getTensorIOMode(name) ==
                nvinfer1::TensorIOMode::kOUTPUT) {
                require(output_name_.empty(), "engine has multiple outputs");
                output_name_ = name;
            }
        }
        require(
            !input_name_.empty() && !output_name_.empty(),
            "engine has no valid input/output");
        require(
            engine_->getTensorDataType(input_name_.c_str()) ==
                    nvinfer1::DataType::kFLOAT &&
                engine_->getTensorDataType(output_name_.c_str()) ==
                    nvinfer1::DataType::kFLOAT,
            "engine tensors must be FP32");

        const nvinfer1::Dims input_dims =
            engine_->getTensorShape(input_name_.c_str());
        const nvinfer1::Dims output_dims =
            engine_->getTensorShape(output_name_.c_str());

        const auto element_count = [this](const nvinfer1::Dims& dims) {
            int64_t count = 1;
            for (int32_t dim = 0; dim < dims.nbDims; ++dim) {
                require(dims.d[dim] > 0, "dynamic shapes are not supported");
                count *= dims.d[dim];
            }
            return count;
        };

        input_elements_ = element_count(input_dims);
        const int64_t output_elements = element_count(output_dims);
        output_shape_.assign(
            output_dims.d,
            output_dims.d + output_dims.nbDims);

        check_cuda(
            cudaMalloc(&input_device_, static_cast<size_t>(input_elements_) * sizeof(float)),
            "cudaMalloc failed");
        check_cuda(
            cudaMalloc(
                &output_device_,
                static_cast<size_t>(output_elements) * sizeof(float)),
            "cudaMalloc failed");
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate failed");
        output_host_.resize(static_cast<size_t>(output_elements));
    } catch (...) {
        release();
        throw;
    }
}

TensorRtInference::~TensorRtInference() {
    release();
}

void TensorRtInference::release() noexcept {
    if (stream_) {
        (void)cudaStreamDestroy(stream_);
    }
    if (input_device_) {
        (void)cudaFree(input_device_);
    }
    if (output_device_) {
        (void)cudaFree(output_device_);
    }
    delete context_;
    delete engine_;
    delete runtime_;
    stream_ = nullptr;
    input_device_ = nullptr;
    output_device_ = nullptr;
    context_ = nullptr;
    engine_ = nullptr;
    runtime_ = nullptr;
}

cv::Mat TensorRtInference::infer(const cv::Mat& letterboxed_rgb) {
    const cv::Mat blob = cv::dnn::blobFromImage(
        letterboxed_rgb,
        1.0 / 255.0,
        letterboxed_rgb.size(),
        cv::Scalar(),
        false,
        false,
        CV_32F);
    require(
        blob.total() == static_cast<size_t>(input_elements_),
        "input size does not match engine");

    check_cuda(
        cudaMemcpyAsync(
            input_device_,
            blob.ptr<float>(),
            blob.total() * blob.elemSize(),
            cudaMemcpyHostToDevice,
            stream_),
        "cudaMemcpy failed");
    require(
        context_->setTensorAddress(input_name_.c_str(), input_device_),
        "failed to bind input tensor");
    require(
        context_->setTensorAddress(output_name_.c_str(), output_device_),
        "failed to bind output tensor");
    require(context_->enqueueV3(stream_), "TensorRT inference failed");
    check_cuda(
        cudaMemcpyAsync(
            output_host_.data(),
            output_device_,
            output_host_.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_),
        "cudaMemcpy failed");
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed");

    require(
        output_shape_.size() == 3,
        "TensorRT output must be [1, channels, boxes] or [1, boxes, channels]");

    const int rows = static_cast<int>(output_shape_[1]);
    const int cols = static_cast<int>(output_shape_[2]);
    cv::Mat channels_first;
    if (rows > cols) {
        cv::Mat box_major(rows, cols, CV_32F, output_host_.data());
        cv::Mat transposed;
        cv::transpose(box_major, transposed);
        transposed.copyTo(channels_first);
    } else {
        cv::Mat channel_major(rows, cols, CV_32F, output_host_.data());
        channel_major.copyTo(channels_first);
    }
    return channels_first;
}

#endif  // PNP_USE_TENSORRT

struct YoloDetector::Impl {
    std::vector<std::string> class_names;
    double conf_threshold = 0.25;
    double nms_threshold = 0.45;
    int input_width = 640;
    int input_height = 640;
#ifdef PNP_USE_LIBTORCH
    torch::jit::script::Module module;
#endif
    cv::dnn::Net net;
#ifdef PNP_USE_TENSORRT
    std::unique_ptr<TensorRtInference> tensorrt;
#endif
};

YoloDetector::YoloDetector(
    const std::string& model_path,
    std::vector<std::string> class_names,
    double conf_threshold,
    double nms_threshold,
    int input_size,
    const std::string& runtime)
    : impl_(std::make_unique<Impl>()) {
    if (model_path.empty()) {
        throw std::invalid_argument("empty model path");
    }
    if (input_size <= 0) {
        throw std::invalid_argument("input_size must be positive");
    }
    impl_->class_names = std::move(class_names);
    impl_->conf_threshold = conf_threshold;
    impl_->nms_threshold = nms_threshold;
    impl_->input_width = input_size;
    impl_->input_height = input_size;

    const std::string lower_runtime = util::lower_ascii(runtime);
    const std::string lower_model = util::lower_ascii(model_path);
    const bool tensorrt_path =
        util::ends_with(lower_model, ".engine") || util::ends_with(lower_model, ".trt");
    const bool onnx_path = util::ends_with(lower_model, ".onnx");
    const bool torch_path =
        util::ends_with(lower_model, ".torchscript") || util::ends_with(lower_model, ".pt");

    const bool use_tensorrt =
        lower_runtime == "tensorrt" || lower_runtime == "trt" ||
        (lower_runtime == "auto" && tensorrt_path);
    if (use_tensorrt) {
#ifdef PNP_USE_TENSORRT
        impl_->tensorrt = std::make_unique<TensorRtInference>(model_path);
#else
        throw std::invalid_argument("TensorRT support is not compiled");
#endif
        return;
    }

    const bool use_libtorch =
        lower_runtime == "libtorch" ||
        (lower_runtime == "auto" && (torch_path || (!onnx_path && !tensorrt_path)));
    if (use_libtorch) {
#ifdef PNP_USE_LIBTORCH
        try {
            impl_->module = torch::jit::load(model_path);
            impl_->module.eval();
        } catch (const std::exception& error) {
            throw std::invalid_argument(
                "failed to load TorchScript model: " + model_path +
                ": " + error.what());
        }
#else
        throw std::invalid_argument("LibTorch support is not compiled");
#endif
        return;
    }

    const bool use_opencv = lower_runtime == "onnx" || onnx_path;
    if (!use_opencv) {
        throw std::invalid_argument("unknown runtime: " + runtime);
    }
    impl_->net = cv::dnn::readNetFromONNX(model_path);
    if (impl_->net.empty()) {
        throw std::invalid_argument(
            "failed to load ONNX model: " + model_path);
    }
}

YoloDetector::~YoloDetector() = default;

std::vector<Detection> YoloDetector::detectRgb(
    const cv::Mat& rgb,
    const std::vector<int>& target_classes) {
    return detectRgb(rgb, impl_->conf_threshold, target_classes);
}

std::vector<Detection> YoloDetector::detectRgb(
    const cv::Mat& rgb,
    double min_conf,
    const std::vector<int>& target_classes) {
    if (rgb.empty()) {
        throw std::invalid_argument("empty image");
    }
    if (rgb.type() != CV_8UC3) {
        throw std::invalid_argument("expected CV_8UC3 image");
    }

    int pad_x = 0;
    int pad_y = 0;
    float scale_x = 1.0F;
    float scale_y = 1.0F;
    const cv::Mat letterboxed =
        letterbox(rgb, impl_->input_width, impl_->input_height, pad_x, pad_y, scale_x, scale_y);

    cv::Mat output;
#ifdef PNP_USE_TENSORRT
    if (impl_->tensorrt != nullptr) {
        output = impl_->tensorrt->infer(letterboxed);
    } else
#endif
    {
#ifdef PNP_USE_LIBTORCH
        if (impl_->net.empty()) {
            torch::NoGradGuard no_grad;
            torch::Tensor tensor = torch::from_blob(
                                       letterboxed.data,
                                       {1, impl_->input_height, impl_->input_width, 3},
                                       torch::kUInt8)
                                       .clone();
            tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat32).div_(255.0);
            torch::Tensor result = impl_->module.forward({tensor}).toTensor().contiguous();
            if (result.dim() != 3 || result.size(0) != 1) {
                throw std::runtime_error(
                    "unsupported TorchScript output shape");
            }
            output = cv::Mat(
                         static_cast<int>(result.size(1)),
                         static_cast<int>(result.size(2)),
                         CV_32F,
                         const_cast<float*>(result.data_ptr<float>()))
                         .clone();
        } else
#endif
        {
            const cv::Mat blob = cv::dnn::blobFromImage(
                letterboxed,
                1.0 / 255.0,
                cv::Size(impl_->input_width, impl_->input_height),
                cv::Scalar(),
                false,
                false,
                CV_32F);

            impl_->net.setInput(blob);
            output = impl_->net.forward();
        }
    }

    cv::Mat channels_first;
    if (output.dims == 3) {
        const int dim1 = output.size[1];
        const int dim2 = output.size[2];
        const int shape[] = {dim1, dim2};
        if (dim1 > dim2) {
            output = output.reshape(1, 2, shape);
            cv::transpose(output, channels_first);
        } else {
            channels_first = output.reshape(1, 2, shape);
        }
    } else if (output.dims == 2) {
        channels_first = output;
    } else {
        throw std::runtime_error("unsupported model output shape");
    }

    const int num_channels = channels_first.rows;
    const int num_boxes = channels_first.cols;
    if (num_channels < 5 || num_boxes <= 0) {
        throw std::runtime_error("invalid model output dimensions");
    }

    const float* data = channels_first.ptr<float>();
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(num_boxes));

    for (int i = 0; i < num_boxes; ++i) {
        const float cx = data[i];
        const float cy = data[num_boxes + i];
        const float bw = data[2 * num_boxes + i];
        const float bh = data[3 * num_boxes + i];

        if (!std::isfinite(cx) || !std::isfinite(cy) ||
            !std::isfinite(bw) || !std::isfinite(bh) ||
            bw <= 0.0F || bh <= 0.0F) {
            continue;
        }

        float best_score = 0.0F;
        int best_cls = -1;
        for (int c = 4; c < num_channels; ++c) {
            const float score = data[c * num_boxes + i];
            if (score < static_cast<float>(min_conf) || score <= best_score) {
                continue;
            }
            best_score = score;
            best_cls = c - 4;
        }

        if (best_cls < 0) {
            continue;
        }
        if (!target_classes.empty() &&
            std::find(target_classes.begin(), target_classes.end(), best_cls) ==
                target_classes.end()) {
            continue;
        }

        candidates.push_back(Candidate{
            cx - bw * 0.5F,
            cy - bh * 0.5F,
            cx + bw * 0.5F,
            cy + bh * 0.5F,
            best_score,
            best_cls});
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    boxes.reserve(candidates.size());
    scores.reserve(candidates.size());
    class_ids.reserve(candidates.size());

    for (const Candidate& candidate : candidates) {
        boxes.push_back(cv::Rect(
            static_cast<int>(std::floor(candidate.x1)),
            static_cast<int>(std::floor(candidate.y1)),
            static_cast<int>(std::ceil(candidate.x2 - candidate.x1)),
            static_cast<int>(std::ceil(candidate.y2 - candidate.y1))));
        scores.push_back(candidate.score);
        class_ids.push_back(candidate.cls);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(
        boxes,
        scores,
        static_cast<float>(min_conf),
        static_cast<float>(impl_->nms_threshold),
        indices);

    std::vector<Detection> detections;
    detections.reserve(indices.size());
    for (int index : indices) {
        const Candidate& candidate = candidates[static_cast<size_t>(index)];
        const float x1 = std::clamp((candidate.x1 - pad_x) * scale_x, 0.0F, static_cast<float>(rgb.cols));
        const float y1 = std::clamp((candidate.y1 - pad_y) * scale_y, 0.0F, static_cast<float>(rgb.rows));
        const float x2 = std::clamp((candidate.x2 - pad_x) * scale_x, 0.0F, static_cast<float>(rgb.cols));
        const float y2 = std::clamp((candidate.y2 - pad_y) * scale_y, 0.0F, static_cast<float>(rgb.rows));

        std::string name =
            candidate.cls >= 0 &&
                    static_cast<size_t>(candidate.cls) < impl_->class_names.size()
                ? impl_->class_names[static_cast<size_t>(candidate.cls)]
                : std::to_string(candidate.cls);

        detections.push_back(Detection{
            cv::Vec4f(x1, y1, x2, y2),
            candidate.cls,
            candidate.score,
            std::move(name)});
    }
    return detections;
}

std::vector<Detection> YoloDetector::detectBgr(
    const cv::Mat& bgr,
    const std::vector<int>& target_classes) {
    return detectBgr(bgr, impl_->conf_threshold, target_classes);
}

std::vector<Detection> YoloDetector::detectBgr(
    const cv::Mat& bgr,
    double min_conf,
    const std::vector<int>& target_classes) {
    if (bgr.empty()) {
        throw std::invalid_argument("empty image");
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return detectRgb(rgb, min_conf, target_classes);
}

std::vector<Detection> keep_highest_confidence(std::vector<Detection> detections) {
    if (detections.empty()) {
        return {};
    }
    const auto best =
        std::max_element(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
            return a.conf < b.conf;
        });
    return {*best};
}

}  // namespace pnp_vision::detection
