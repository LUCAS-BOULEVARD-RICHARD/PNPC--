#include <opencv2/core.hpp>

#include <vector>

namespace pnp_vision::tasks {

std::vector<cv::Point3f> make_object_points(
    int pattern_width,
    int pattern_height,
    double square) {
    std::vector<cv::Point3f> points;
    points.reserve(static_cast<size_t>(pattern_width * pattern_height));
    for (int y = 0; y < pattern_height; ++y) {
        for (int x = 0; x < pattern_width; ++x) {
            points.push_back(cv::Point3f(
                static_cast<float>(x * square),
                static_cast<float>(y * square),
                0.0F));
        }
    }
    return points;
}

}  // namespace pnp_vision::tasks
