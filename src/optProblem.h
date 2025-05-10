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
    Eigen::MatrixXd I;                  // Grayscale images (pixels × lights)
    Eigen::VectorXd rho;                // Albedo (per-pixel)
    std::vector<Eigen::Vector3d> light_positions; // Known 3D positions of lights
    std::vector<Eigen::Matrix3d> J_all_pixels;    // Jacobians for all pixels
    std::vector<Eigen::Vector3d> light_dirs; // [pixel][light]
    std::vector<double> light_distances;           // [pixel][light]
};

// Ceres cost functor for depth optimization
struct DepthCostFunctor {
    DepthCostFunctor(
                    double I_ji, double rho_j, double phi_i,
                    const Eigen::Matrix3d& J,
                    const double light_distance,
                    const Eigen::Vector3d& light_dir,
                    const double anisotropy);

    template <typename T>
    bool operator()(
        const T* const z_j,
        const T* const z_right,
        const T* const z_bottom,
        T* residual) const;

private:
    double I_ji, rho_j, phi_i;
    const Eigen::Matrix3d& J;
    double light_distance;
    Eigen::Vector3d light_dir;
    double anisotropy;
    static int debug_count;
};



// Function to optimize the depth map
void optimizeDepthMap(Eigen::VectorXd& z, double* z_p, const PrecomputedData& data);
void runFullOptimization(PrecomputedData& data);


#endif // OPT_PROBLEM_H