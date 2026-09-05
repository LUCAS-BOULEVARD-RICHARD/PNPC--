#include <stdexcept>
#include <string>
#include <utility>

namespace pnp_vision::tasks {

std::pair<int, int> parse_pattern(const std::string& pattern) {
    const size_t x = pattern.find('x');
    if (x == std::string::npos) {
        throw std::runtime_error("invalid pattern, expected WxH");
    }
    return {
        std::stoi(pattern.substr(0, x)),
        std::stoi(pattern.substr(x + 1))};
}

}  // namespace pnp_vision::tasks
