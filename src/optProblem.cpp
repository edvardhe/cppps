#include "optProblem.h"

#include <filesystem>
#include <fstream>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <utility>
#include <vector>
#include <c++/12/bits/std_thread.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include "depthMapHandler.h"

/**
 * Callback for saving depth map images during optimization
 */
class DepthMapCallback : public ceres::IterationCallback {
public:
    DepthMapCallback(
        int width,
        int height,
        double* depth_data,
        size_t depth_size
    ) : width_(width),
        height_(height),
        depth_data_(depth_data),
        depth_size_(depth_size),
        iteration_count_(0) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
        // Save the current depth map
        output_dir_ = "/home/edvard/dev/projects/cppPS/depthMapIterations";

        std::string output_name = "iter_" + std::to_string(iteration_count_);

        Eigen::MatrixXd depth_map(Eigen::Map<Eigen::MatrixXd>(depth_data_, height_, width_));

        //cv::Mat debug;
        //cv::eigen2cv(depth_map.eval(), debug);
        //cv::normalize(debug,debug,0,255,cv::NORM_MINMAX);


        depth::saveDepthMap(depth_map.eval(), output_dir_, output_name);

        // Increment iteration counter
        iteration_count_++;

        // Continue optimization
        return ceres::SOLVER_CONTINUE;
    }
private:
    // Image dimensions
    int width_;
    int height_;

    // Reference to depth parameters
    double* depth_data_;
    size_t depth_size_;

    // Output options
    std::string output_dir_;

    // Iteration counter
    int iteration_count_;
};

// Implement constructor
DepthCostFunctor::DepthCostFunctor(
    double I_ji, double rho_j, double phi_i,
    const Eigen::Matrix3d& J,
    const double light_distance,
    const Eigen::Vector3d& light_dir,
    const double anisotropy)
: I_ji(I_ji), rho_j(rho_j), phi_i(phi_i), J(J),
  light_distance(light_distance), light_dir(light_dir),
  anisotropy(anisotropy) {}

// 1. Photometric residual only
struct PhotometricResidual {
    PhotometricResidual(
    const double I_ji, const double rho_j,
    const Eigen::Matrix3d& J,
    const double light_distance,
    const Eigen::Vector3d& light_dir)
: I_ji(I_ji), rho_j(rho_j), J(J),
  light_distance(light_distance), light_dir(light_dir){}

    template <typename T>
    bool operator()(const T* const z_j,
                    const T* const z_right,
                    const T* const z_bottom,
                    T* residual) const {
        // --- Gradient Calculation with Boundary Handling ---
        T dz_x, dz_y;

        // center difference
        //dz_x = (*z_right - *z_left) / T(2.0);
        //dz_y = (*z_bottom - *z_top) / T(2.0);

        // forward difference
        dz_x = (*z_j - *z_right);
        dz_y = (*z_j - *z_bottom);

        Eigen::Matrix<T, 3, 1> grad_z_neg1(dz_x, dz_y, T(-1.0));
        Eigen::Matrix<T, 3, 1> n = J.transpose() * grad_z_neg1;
        n.normalize();

        // Compute lighting
        Eigen::Matrix<T, 3, 1> light_dir_t = light_dir.cast<T>();
        T distance = T(light_distance);

        T falloff = T(1.0) / (distance * distance);
        Eigen::Matrix<T, 3, 1> s = falloff * light_dir_t;
        Eigen::Matrix<T, 1, 3> JsT = (J * s).transpose();
        T incoming_light = JsT * grad_z_neg1;
        T light_estimate = ceres::fmax(incoming_light,T(0.0));
        T albedo_adjusted_estimate = light_estimate * (rho_j);

        // Compute residual
        residual[0] = (albedo_adjusted_estimate - T(I_ji)) / T(rho_j);
        return true;
    }
private:
    double I_ji;
    double rho_j;
    double phi_i;
    Eigen::Matrix3d J;
    double light_distance;
    Eigen::Vector3d light_dir;
};

// 2. Smoothness residual only
struct SmoothnessResidual {
    SmoothnessResidual() {}

    template <typename T>
    bool operator()(const T* const z_center,
                    const T* const z_right,
                    const T* const z_left,
                    const T* const z_top,
                    const T* const z_bottom,
                    T* residual) const {
        // First derivatives (gradient)
        T dz_x = (*z_right - *z_left) / T(2.0);
        T dz_y = (*z_bottom - *z_top) / T(2.0);

        T laplacian = *z_right + *z_left + *z_top + *z_bottom - 4.0 * *z_center;
        T curve_penalty = laplacian;
        // Roughness penalty
        T weight = T(0);
        residual[0] = curve_penalty * weight;
        return true;
    }
};

struct depthResidual {
    depthResidual() {}

    template <typename T>
    bool operator()(const T* const z_j,
                    T* residual) const {
        // Depth prior
        T depth_prior = T(2.147);
        T penalty = ceres::fmax(T(0.0), depth_prior - *z_j);
        T depth_prior_penalty = T(100.0) * penalty;
        residual[0] = depth_prior_penalty;
        return true;
    }
};


// Main function for depth optimization step
void optimizeDepthMap(Eigen::VectorXd& z, double* z_p, const PrecomputedData& data) {
    
    // Add debug prints at the start of the function
    std::cout << "Matrix sizes:" << std::endl;
    std::cout << "I: " << data.I.rows() << "x" << data.I.cols() << std::endl;
    std::cout << "light_positions size: " << data.light_positions.size() << std::endl;

    // Print value ranges
    std::cout << "Value ranges:" << std::endl;
    std::cout << "I range: " << data.I.minCoeff() << " to " << data.I.maxCoeff() << std::endl;
    std::cout << "rho range: " << data.rho.minCoeff() << " to " << data.rho.maxCoeff() << std::endl;
    
    // Print a few light positions
    std::cout << "First light position: " << data.light_positions[0].transpose() << std::endl;
    
    // Print camera matrix
    std::cout << "Camera matrix K:\n" << data.K << std::endl;

    ceres::Problem problem;
    int image_width = data.width;
    int image_height = data.height;

    // ceres::CostFunction* depth_cost_function =
    // new ceres::AutoDiffCostFunction<depthResidual, 1, 1>(
    //     new depthResidual()
    // );
    //
    // ceres::CostFunction* smooth_cost_function =
    // new ceres::AutoDiffCostFunction<SmoothnessResidual, 1, 1, 1, 1, 1, 1>(
    //     new SmoothnessResidual()
    // );


    // Add residuals for all pixels and lights
    for (int j = 0; j < data.I.rows(); ++j) { // For each pixel
        int x = j / image_height; // column major (И shapes)
        int y = j % image_height;

        //problem.AddResidualBlock(
        //    depth_cost_function   ,
        //    nullptr,
        //    &z(j)        // Current depth
        //);


        // Skip boundary pixels (no neighbors exist)
        if (x == image_width - 1 || y == image_height - 1) {
            continue;
        }

        // Neighbor indices (safe, since we skipped boundaries)
        int j_right = j + image_height;
        int j_left = j - image_height;
        int j_bottom = j + 1;
        int j_top = j - 1;

        for (int i = 0; i < data.I.cols(); ++i) { // For each light
            int idx = j * data.I.cols() + i;

            ceres::CostFunction* photometric_cost_function =
                new ceres::AutoDiffCostFunction<PhotometricResidual, 1, 1, 1, 1>(
                    new PhotometricResidual(
                        data.I(j, i), data.rho(j),
                        data.J_all_pixels[j],
                        data.light_distances[idx],
                        data.light_dirs[idx]
                    )
                );

            // Add residual block with all 5 parameters
            problem.AddResidualBlock(
                photometric_cost_function   ,
                new ceres::CauchyLoss(30),
                &z(j),        // Current depth
                &z(j_right),  // Right neighbor
                &z(j_bottom)  // Bottom neighbor
            );
        }

        if (x == 0 || y == 0) {
            continue;
        }

        //problem.AddResidualBlock(
        //    smooth_cost_function   ,
        //    nullptr,
        //    &z(j),
        //    &z(j_right),  // Right neighbor
        //    &z(j_left),
        //    &z(j_top),
        //    &z(j_bottom)
        //);
    }

    for (int j = 0; j < z.size(); ++j) {

        int x = j / image_height;
        int y = j % image_height;

        if (x == image_width - 1 && y == image_height - 1) continue;

        //problem.SetParameterLowerBound(&z(j), 0, 2.147);
        //problem.SetParameterUpperBound(&z(j), 0, 2.16);

        if (x == 0 || x == image_width - 1 || y == 0 || y == image_height - 1) {

            //if (x == 0 && y == 0) continue;
            //if (x == 0 && y == image_height - 1) continue;
            //if (x == image_width - 1 && y == 0) continue;
            problem.SetParameterBlockConstant(&z(j));
            // OR: Set to a neighbor's value
            // if (x == image_width - 1) z(j) = z(j - 1);
        }
    }


    // Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.preconditioner_type = ceres::JACOBI;
    options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 5;
    options.max_linear_solver_iterations = 1;
    options.function_tolerance = 1e-9;
    options.gradient_tolerance = 1e-7;
    options.parameter_tolerance = 4e-9;
    options.jacobi_scaling = true;
    options.use_inner_iterations = false;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    // Take smaller initial steps
    options.initial_trust_region_radius = 1.0;  // default 10
    options.max_trust_region_radius = 20.0;    // Limit how large the trust region can grow


    // // Line search specific options
    // options.minimizer_type = ceres::LINE_SEARCH;
    // options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;  // or ceres::STEEPEST_DESCENT, ceres::NONLINEAR_CONJUGATE_GRADIENT
    // options.line_search_type = ceres::WOLFE;           // or ceres::ARMIJO
    // options.line_search_interpolation_type = ceres::CUBIC;
    //
    // // Line search parameters
    // options.max_lbfgs_rank = 20;  // For LBFGS direction
    // options.use_approximate_eigenvalue_bfgs_scaling = true;
    // options.line_search_sufficient_function_decrease = 1e-4;
    // options.line_search_sufficient_curvature_decrease = 0.9;
    // options.max_line_search_step_contraction = 1e-3;
    // options.min_line_search_step_contraction = 0.6;
    // options.max_num_line_search_step_size_iterations = 20;
    // options.max_num_line_search_direction_restarts = 5;
    // options.line_search_sufficient_curvature_decrease = 0.3;

    options.num_threads = 16;

    // Create and add the callback for saving depth maps
    options.update_state_every_iteration = true;
    options.logging_type = ceres::PER_MINIMIZER_ITERATION;
    auto* callback = new DepthMapCallback(
        data.width, data.height, z.data(), z.size()
    );
    options.callbacks.push_back(callback);


    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    
    std::cout << "Initial cost: " << summary.initial_cost << std::endl;
    std::cout << "Final cost: " << summary.final_cost << std::endl;
    std::cout << "Termination: " << summary.termination_type << std::endl;
    std::cout << summary.FullReport() << "\n";

    // Check if optimization actually improved the solution
    if (summary.final_cost >= summary.initial_cost) {
        std::cout << "Warning: Optimization did not decrease cost!" << std::endl;
    }
}


