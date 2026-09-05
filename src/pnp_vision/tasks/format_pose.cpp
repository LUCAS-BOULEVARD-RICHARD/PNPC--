#include "pnp_vision/geometry/pnp.hpp"

#include <iomanip>
#include <sstream>
#include <string>

namespace pnp_vision::tasks {

std::string format_pose(const geometry::PnPPose& pose) {
    const cv::Vec3d euler = geometry::rvec_to_euler_zyx(pose.rvec());
    std::ostringstream text;
    text << std::fixed << std::setprecision(3)
         << "X=" << pose.x() << " Y=" << pose.y() << " Z=" << pose.z()
         << " dist=" << pose.distance() << " Roll=" << std::setw(7)
         << euler[0] << " Pitch=" << std::setw(7) << euler[1]
         << " Yaw=" << std::setw(7) << euler[2];
    return text.str();
}

}  // namespace pnp_vision::tasks
