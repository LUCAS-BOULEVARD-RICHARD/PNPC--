#ifndef PNP_VISION_YAML_HPP
#define PNP_VISION_YAML_HPP

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// yaml-cpp 的线程安全单例封装。嵌套配置使用点路径访问，
// 例如 "detection.confidence"。
class ConfigManager {
public:
    static ConfigManager& getInstance() {
        static ConfigManager instance;
        return instance;
    }

    void init(const std::string& config_path) {
        std::lock_guard<std::mutex> lock(mtx_);
        config_ = YAML::LoadFile(config_path);
        path_ = config_path;
    }

    template <typename T>
    T get(const std::string& key, T default_value) {
        std::lock_guard<std::mutex> lock(mtx_);
        try {
            const YAML::Node node = findNode(key);
            if (node && node.IsDefined() && !node.IsNull()) {
                return node.as<T>();
            }
        } catch (...) {
            // 缺失或无法解析的值回退到默认值。
        }
        return default_value;
    }

    template <typename T>
    std::optional<T> getOptional(const std::string& key) {
        std::lock_guard<std::mutex> lock(mtx_);
        try {
            const YAML::Node node = findNode(key);
            if (!node || node.IsNull()) {
                return std::nullopt;
            }
            if (node.IsScalar()) {
                std::string text = node.Scalar();
                std::transform(
                    text.begin(),
                    text.end(),
                    text.begin(),
                    [](unsigned char ch) {
                        return static_cast<char>(std::tolower(ch));
                    });
                if (text.empty() || text == "none" ||
                    text == "null" || text == "~") {
                    return std::nullopt;
                }
            }
            return node.as<T>();
        } catch (...) {
            return std::nullopt;
        }
    }

    template <typename T>
    std::vector<T> getVector(const std::string& key) {
        std::lock_guard<std::mutex> lock(mtx_);
        try {
            const YAML::Node node = findNode(key);
            if (node && node.IsSequence()) {
                return node.as<std::vector<T>>();
            }
        } catch (...) {
            // 无效的列表值按空列表处理。
        }
        return {};
    }

private:
    ConfigManager() = default;
    ~ConfigManager() = default;
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

    YAML::Node findNode(const std::string& key) {
        YAML::Node current = YAML::Clone(config_);
        std::string segment;
        std::istringstream stream(key);
        while (std::getline(stream, segment, '.')) {
            if (current[segment]) {
                current = current[segment];
            } else {
                return YAML::Node();
            }
        }
        return current;
    }

    YAML::Node config_;
    std::string path_;
    std::mutex mtx_;
};

#endif  // PNP_VISION_YAML_HPP
