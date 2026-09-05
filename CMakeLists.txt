cmake_minimum_required(VERSION 3.16)

project(PNPCpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d videoio highgui dnn)
find_package(Threads REQUIRED)

add_library(pnp_vision STATIC
    src/pnp_vision/camera/capture.cpp
    src/pnp_vision/camera/hik.cpp
    src/pnp_vision/config/config.cpp
    src/pnp_vision/detection/detector.cpp
    src/pnp_vision/geometry/calibration.cpp
    src/pnp_vision/geometry/gimbal.cpp
    src/pnp_vision/geometry/pnp.cpp
    src/pnp_vision/tasks/cli.cpp
    src/pnp_vision/tasks/exposure.cpp
    src/pnp_vision/tasks/format_pose.cpp
    src/pnp_vision/tasks/make_object_points.cpp
    src/pnp_vision/tasks/parse_pattern.cpp
    src/pnp_vision/tasks/runner.cpp
    src/pnp_vision/visualization/drawing.cpp
    src/pnp_vision/visualization/hud.cpp
    src/pnp_vision/visualization/tracking.cpp
)

find_package(yaml-cpp REQUIRED)

target_include_directories(pnp_vision PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(pnp_vision PUBLIC ${OpenCV_LIBS} yaml-cpp Threads::Threads)

set(PNP_TENSORRT_DIR
    "/usr"
    CACHE PATH "Path prefix to the TensorRT SDK")
option(PNP_ENABLE_TENSORRT "Build with TensorRT when the SDK is available" ON)
option(PNP_ENABLE_LIBTORCH "Build with LibTorch when it is available" ON)
option(PNP_ENABLE_HIKVISION "Build with the Hikvision MVS SDK when available" ON)

if(PNP_TENSORRT_DIR)
    list(PREPEND CMAKE_PREFIX_PATH "${PNP_TENSORRT_DIR}")
endif()

find_path(PNP_TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS "${PNP_TENSORRT_DIR}/include"
          /usr/include/x86_64-linux-gnu
          /usr/include
          /usr/local/tensorrt/include
    NO_DEFAULT_PATH)
find_library(PNP_TENSORRT_LIBRARY
    NAMES nvinfer
    HINTS "${PNP_TENSORRT_DIR}/lib"
          "${PNP_TENSORRT_DIR}/lib64"
          /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH)
find_library(PNP_TENSORRT_PLUGIN_LIBRARY
    NAMES nvinfer_plugin
    HINTS "${PNP_TENSORRT_DIR}/lib"
          "${PNP_TENSORRT_DIR}/lib64"
          /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH)

if(PNP_ENABLE_TENSORRT
   AND PNP_TENSORRT_INCLUDE_DIR
   AND PNP_TENSORRT_LIBRARY)
    find_package(CUDAToolkit REQUIRED)
    add_library(pnp_tensorrt INTERFACE)
    target_include_directories(pnp_tensorrt INTERFACE
        ${PNP_TENSORRT_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(pnp_tensorrt INTERFACE
        ${PNP_TENSORRT_LIBRARY}
        CUDA::cudart
    )
    if(PNP_TENSORRT_PLUGIN_LIBRARY)
        target_link_libraries(pnp_tensorrt INTERFACE
            ${PNP_TENSORRT_PLUGIN_LIBRARY}
        )
    endif()
    target_link_libraries(pnp_vision PUBLIC pnp_tensorrt)
    target_compile_definitions(pnp_vision PUBLIC PNP_USE_TENSORRT)
    message(STATUS "TensorRT enabled: ${PNP_TENSORRT_INCLUDE_DIR}")
else()
    message(STATUS "TensorRT disabled or not found; use onnx/libtorch runtime")
endif()

set(PNP_TORCH_DIR
    ""
    CACHE PATH "Path to the LibTorch package (empty = search system paths)")

if(PNP_TORCH_DIR)
    set(Torch_DIR "${PNP_TORCH_DIR}/share/cmake/Torch")
endif()

if(PNP_ENABLE_LIBTORCH)
    find_package(Torch QUIET)
endif()

if(PNP_ENABLE_LIBTORCH AND Torch_FOUND)
    target_compile_definitions(pnp_vision PUBLIC PNP_USE_LIBTORCH)
    target_include_directories(pnp_vision PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(pnp_vision PUBLIC ${TORCH_LIBRARIES})
    if(PNP_TORCH_DIR)
        list(APPEND CMAKE_BUILD_RPATH "${PNP_TORCH_DIR}/lib")
    endif()
    message(STATUS "LibTorch enabled: ${Torch_DIR}")
else()
    message(STATUS "LibTorch disabled or not found; use tensorrt/onnx runtime")
endif()

set(PNP_MVS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../PNP/third_party/MVS"
    CACHE PATH "Path to the Hikvision MVS SDK")

if(PNP_ENABLE_HIKVISION
   AND EXISTS "${PNP_MVS_DIR}/include/MvCameraControl.h"
   AND EXISTS "${PNP_MVS_DIR}/lib/64/libMvCameraControl.so")
    add_library(mvs INTERFACE)
    target_include_directories(mvs INTERFACE "${PNP_MVS_DIR}/include")
    target_link_directories(mvs INTERFACE "${PNP_MVS_DIR}/lib/64")
    target_link_libraries(mvs INTERFACE MvCameraControl)

    target_compile_definitions(pnp_vision PUBLIC PNP_HAS_HIKVISION)
    target_link_libraries(pnp_vision PUBLIC mvs)
    list(APPEND CMAKE_BUILD_RPATH "${PNP_MVS_DIR}/lib/64")
    message(STATUS "Hikvision MVS SDK enabled: ${PNP_MVS_DIR}")
else()
    message(STATUS "Hikvision disabled or SDK not found; use laptop/usb camera")
endif()

function(add_pnp_task target_name task_macro)
    add_executable(${target_name} src/main.cpp)
    target_compile_definitions(${target_name} PRIVATE ${task_macro})
    target_link_libraries(${target_name} PRIVATE pnp_vision)
endfunction()

add_pnp_task(task1_yolo_detection PNP_TASK_DETECTION)
add_pnp_task(task2_pnp_distance PNP_TASK_DISTANCE)
add_pnp_task(task3_pnp_pose PNP_TASK_POSE)
add_pnp_task(task4_pnp_visualize PNP_TASK_VISUALIZE)
add_pnp_task(calibrate_camera PNP_TASK_CALIBRATION)
add_pnp_task(adjust_exposure PNP_TASK_EXPOSURE)
