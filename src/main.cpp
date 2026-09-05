#include "pnp_vision/tasks/runner.hpp"

#include <stdexcept>

int main(int argc, char** argv) {
#if defined(PNP_TASK_DETECTION)
    return pnp_vision::tasks::main_task(
        argc, argv, pnp_vision::tasks::TaskKind::Detection);
#elif defined(PNP_TASK_DISTANCE)
    return pnp_vision::tasks::main_task(
        argc, argv, pnp_vision::tasks::TaskKind::Distance);
#elif defined(PNP_TASK_POSE)
    return pnp_vision::tasks::main_task(
        argc, argv, pnp_vision::tasks::TaskKind::Pose);
#elif defined(PNP_TASK_VISUALIZE)
    return pnp_vision::tasks::main_task(
        argc, argv, pnp_vision::tasks::TaskKind::Visualize);
#elif defined(PNP_TASK_CALIBRATION)
    return pnp_vision::tasks::main_task(
        argc, argv, pnp_vision::tasks::TaskKind::Calibration);
#elif defined(PNP_TASK_EXPOSURE)
    return pnp_vision::tasks::main_task(
        argc, argv, pnp_vision::tasks::TaskKind::Exposure);
#else
    throw std::invalid_argument("main.cpp: no task kind macro defined");
#endif
}
