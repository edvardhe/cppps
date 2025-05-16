#ifndef OPT_PROBLEM_H
#define OPT_PROBLEM_H

#include <Eigen/Dense>
#include <vector>
#include <ceres/ceres.h>

// Precomputed data (from previous steps)
struct PrecomputedData {
    int start_x;                        // Starting x-coordinate
    int start_y;                        // Starting y-coordinate
    int width;                          // Image width
    int height;                         // Image height
    Eigen::Matrix3d K;                  // Camera intrinsics
    Eigen::Matrix3d Kinv;               // Inverse of camera intrinsics
    Eigen::Matrix3d Kinv_t;             // Transpose of inverse of camera intrinsics
    Eigen::MatrixXd I;                  // Grayscale images (pixels × lights)
    std::vector<Eigen::Vector3d> light_positions; // Known 3D positions of lights
    std::vector<Eigen::Vector3d> geometric_terms;    // geometric term for all pixels
    std::vector<Eigen::Vector3d> light_dirs; // [pixel][light]
    std::vector<double> light_distances;           // [pixel][light]
};

// Function to optimize the depth map
void optimizeDepthMap(Eigen::VectorXd& z, Eigen::VectorXd& rho, const PrecomputedData& data, int iteration_count);
void optimizeAlbedo(Eigen::VectorXd &z, Eigen::VectorXd &rho, const PrecomputedData &data, int iteration_count);
void optimizeDepthAndAlbedo(Eigen::VectorXd& z, Eigen::VectorXd& rho, const PrecomputedData& data);



#endif // OPT_PROBLEM_H