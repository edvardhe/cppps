cmake_minimum_required(VERSION 3.10)
project(cppPS LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug) # Ensure the build type is Debug
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g") # Add debugging symbols

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

# Enable CUDA
find_package(CUDA REQUIRED)


# Add your executable
add_executable(cppPS
                src/cppPS.cpp
                src/MatrixProcessing.cpp
                src/optProblem.cpp
        src/depthMapHandler.cpp
        src/depthMapHandler.h
        src/imagePreprocessing.cpp
        src/imagePreprocessing.h
)

# Link OpenCV and set include directories
target_link_libraries(cppPS PRIVATE ${OpenCV_LIBS} glog::glog Ceres::ceres)
target_include_directories(cppPS PRIVATE ${EIGEN3_INCLUDE_DIRS})