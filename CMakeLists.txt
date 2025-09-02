cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(cppPS LANGUAGES CXX CUDA)

# Change this line from Debug to Release for optimization
set(CMAKE_BUILD_TYPE Release)

# Add optimization flags for Release build
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -flto")

# Keep debug flags for when you need debugging
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")


# Find Eigen3 (no linking required for header-only library)
find_package(Eigen3 REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find nlohmann_json
find_package(nlohmann_json 3.2.0 REQUIRED)

# Find glog
find_package(glog REQUIRED)

# Find Ceres Solver
find_package(Ceres REQUIRED)


# Add your executable
add_executable(cppPS
                src/cppPS.cpp
                src/MatrixProcessing.cpp
                src/optProblem.cpp
        src/depthMapHandler.cpp
        src/depthMapHandler.h
)

target_compile_definitions(cppPS PRIVATE
        PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

# Link OpenCV and set include directories
target_link_libraries(cppPS PRIVATE ${OpenCV_LIBS} glog::glog Ceres::ceres)
target_include_directories(cppPS PRIVATE ${EIGEN3_INCLUDE_DIRS})