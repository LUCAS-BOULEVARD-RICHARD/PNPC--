#pragma once

#include "pnp_vision/tasks/cli.hpp"

namespace pnp_vision::camera {

class Capture;

}  // namespace pnp_vision::camera

namespace pnp_vision::tasks {

struct ExposureControl {
    double gain = 0.0;
    double gain_min = 0.0;
    double gain_max = 100.0;
    double exposure_us = 8000.0;
    double exposure_min_us = 8000.0;
    double exposure_max_us = 50000.0;
    double gain_step_coarse = 0.25;
    double gain_step_fine = 0.05;
    double exposure_step_coarse = 1.5;
    double exposure_step_fine = 1.1;
};

ExposureControl make_exposure_control(
    const config::ExposureSettings& settings);

double clamp_exposure_us(
    const ExposureControl& control,
    double exposure_us);

double gain_step(const ExposureControl& control, bool coarse);

bool read_exposure_state(
    camera::Capture& cap,
    ExposureControl& control);

bool adjust_brightness(
    camera::Capture& cap,
    ExposureControl& control,
    bool brighter,
    bool coarse);

int run_exposure(const TaskArgs& args);

}  // namespace pnp_vision::tasks
