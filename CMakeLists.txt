cmake_minimum_required(VERSION 3.10)

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(cppPS LANGUAGES CXX)

# Change this line from Debug to Release for optimization
set(CMAKE_BUILD_TYPE Release)

# Add optimization flags for Release build
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -flto")

# Keep debug flags for when you need debugging
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Find Qt6
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui)

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
        src/imagePreprocessing.cpp
        src/imagePreprocessing.h
        src/Interface/MainWindow.h
        src/Interface/MainWindow.cpp
        src/Interface/widgets/PhotoDirectoryWidget.cpp
        src/Interface/widgets/PhotoDirectoryWidget.h
        src/Interface/widgets/CameraParametersWidget.cpp
        src/Interface/widgets/CameraParametersWidget.h
        src/Interface/widgets/IndicatorSpheresWidget.cpp
        src/Interface/widgets/IndicatorSpheresWidget.h
        src/Interface/widgets/ObjectParametersWidget.cpp
        src/Interface/widgets/ObjectParametersWidget.h
        src/Interface/widgets/RegionsOfInterestWidget.cpp
        src/Interface/widgets/RegionsOfInterestWidget.h
)

# Link OpenCV and set include directories
target_link_libraries(cppPS PRIVATE
        ${OpenCV_LIBS}
        glog::glog
        Ceres::ceres
        Qt6::Core
        Qt6::Widgets
        Qt6::Gui
)
target_include_directories(cppPS PRIVATE ${EIGEN3_INCLUDE_DIRS})