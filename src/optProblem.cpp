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

class DepthMapCallback : public ceres::IterationCallback {
public:
    DepthMapCallback(
        int width,
        int height,
        double* depth_data,
        int iteration_count
    ) : width_(width),
        height_(height),
        depth_data_(depth_data),
        iteration_count_(iteration_count) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
        // Save the current depth map
        output_dir_ = "/home/edvard/dev/projects/cppPS/depthMapIterations";

        std::string output_name = "iter_" + std::to_string(iteration_count_);

        Eigen::MatrixXd depth_map(Eigen::Map<Eigen::MatrixXd>(depth_data_, height_, width_));

        //cv::Mat debug;
        //cv::eigen2cv(depth_map.eval(), debug);
        //cv::normalize(debug,debug,0,255,cv::NORM_MINMAX);


        depth::saveDepthMap(depth_map.eval(), output_dir_, output_name);

        // Continue optimization
        return ceres::SOLVER_CONTINUE;
    }
private:
    // Image dimensions
    int width_;
    int height_;

    // Reference to depth parameters
    double* depth_data_;

    // Output options
    std::string output_dir_;

    // Iteration counter
    int iteration_count_;
};

class AlbedoCallback : public ceres::IterationCallback {
public:
    AlbedoCallback(
        int width,
        int height,
        double* depth_data,
        int iteration_count
    ) : width_(width),
        height_(height),
        depth_data_(depth_data),
        iteration_count_(iteration_count) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
        // Save the current depth map
        output_dir_ = "/home/edvard/dev/projects/cppPS/AlbedoIterations";

        std::string output_name = "iter_" + std::to_string(iteration_count_);

        Eigen::MatrixXd albedo_map(Eigen::Map<Eigen::MatrixXd>(depth_data_, height_, width_));

        cv::Mat albedo_image;
        cv::eigen2cv(albedo_map.eval(), albedo_image);
        cv::normalize(albedo_image,albedo_image,0,255,cv::NORM_MINMAX);

        cv::imwrite(output_dir_ + "/" + output_name + ".png", albedo_image);

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

// Photometric residual
struct PhotometricResidual {
    PhotometricResidual(
    const double I_ji, const double rho_j,
    const Eigen::Matrix3d& Kinv,
    const Eigen::Vector3d ray,
    const Eigen::Vector3d light_pos
    )
: I_ji(I_ji), rho_j(rho_j), Kinv(Kinv), ray(ray),
  light_pos(light_pos) {}

    template <typename T>
    bool operator()(const T* const z_j,
                    const T* const z_right,
                    const T* const z_bottom,
                    T* residual) const {
        // --- Gradient Calculation with Boundary Handling ---
        T dz_x, dz_y;

        // forward difference
        dz_x = T(*z_right - *z_j);
        dz_y = T(*z_bottom - *z_j);

        Eigen::Matrix<T, 3, 1> grad_z_neg1(dz_x, dz_y, T(-1.0));
        Eigen::Matrix<T, 3, 1>  x_j_3D = *z_j * ray;

        Eigen::Matrix<T, 3, 1> light_to_point = light_pos - x_j_3D;
        T distance = T(light_to_point.norm());
        Eigen::Matrix<T, 3, 1> s = light_to_point / (distance * distance * distance);

        T incoming_light = T(s.transpose() * Kinv * grad_z_neg1);

        T light_estimate = T(ceres::fmax(incoming_light,T(0.0)));
        T albedo_adjusted_estimate = light_estimate * T(rho_j);

        // Compute residual
        residual[0] = (T(I_ji) - albedo_adjusted_estimate);
        return true;
    }
private:
    double I_ji;
    double rho_j;
    Eigen::Matrix3d Kinv;
    Eigen::Vector3d ray;
    Eigen::Vector3d light_pos;
};

// 1. Photometric residual only
struct AlbedoResidual {
    AlbedoResidual(
    const double I_ji,
    const double z_j,
    const double z_right,
    const double z_bottom,
    const Eigen::Matrix3d& Kinv,
    const Eigen::Vector3d ray,
    const Eigen::Vector3d light_pos)
    : I_ji(I_ji), z_j(z_j), z_right(z_right), z_bottom(z_bottom), Kinv(Kinv),
      ray(ray), light_pos(light_pos){}

    template <typename T>
    bool operator()(const T* const rho_j,
                    T* residual) const {
        // --- Gradient Calculation with Boundary Handling ---
        T dz_x, dz_y;

        // forward difference
        dz_x = T(z_right - z_j);
        dz_y = T(z_bottom - z_j);

        Eigen::Matrix<T, 3, 1> grad_z_neg1(dz_x, dz_y, T(-1.0));
        Eigen::Matrix<T, 3, 1>  x_j_3D = T(z_j) * ray;

        Eigen::Matrix<T, 3, 1> light_to_point = light_pos - x_j_3D;
        T distance = T(light_to_point.norm());
        Eigen::Matrix<T, 3, 1> s = light_to_point / (distance * distance * distance);

        T incoming_light = T(s.transpose() * Kinv * grad_z_neg1);

        T light_estimate = ceres::fmax(incoming_light,T(0.0));
        T albedo_adjusted_estimate = T(light_estimate * (*rho_j));

        // Compute residual
        residual[0] = (T(I_ji) - albedo_adjusted_estimate);
        return true;
    }
private:
    double I_ji;
    double z_j, z_right, z_bottom;
    Eigen::Matrix3d Kinv;
    Eigen::Vector3d ray;
    Eigen::Vector3d light_pos;
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
        T dz_y = (*z_top - *z_bottom) / T(2.0);

        T laplacian = *z_right + *z_left + *z_top + *z_bottom - 4.0 * *z_center;
        T curve_penalty = laplacian;
        // Roughness penalty
        T weight = T(10000);
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

struct problemSetup {
    ceres::Solver::Options options;
    ceres::Problem problem;

    problemSetup() = default;

    problemSetup(const ceres::Solver::Options& options, const ceres::Problem& problem);
};

// Main function for depth optimization step
void optimizeDepthMap(Eigen::VectorXd& z, Eigen::VectorXd& rho, const PrecomputedData& data, int iteration_count) {
    ceres::Problem problem;
    int image_width = data.width;
    int image_height = data.height;



    auto loss = new ceres::CauchyLoss(0.1);
    // Add residuals for all pixels and lights

    ceres::CostFunction* smoothness_cost_function =
        new ceres::AutoDiffCostFunction<SmoothnessResidual, 1, 1, 1, 1, 1, 1>(
            new SmoothnessResidual());

#pragma omp parallel for schedule(dynamic) num_threads(16)
    for (int j = 0; j < data.I.rows(); ++j) { // For each pixel
        int x = j / image_height; // column major (И shapes)
        int y = j % image_height;


        // Skip boundary pixels (no neighbors exist)
        //if (x == image_width - 1 || y == image_height - 1 || x == 0 || y == 0) {
        //    continue;
        //}

        if (x == image_width - 1 || y == image_height - 1) {
            continue;
        }

        // Neighbor indices
        int j_right = j + image_height;
        int j_left = j - image_height;
        int j_bottom = j + 1;
        int j_top = j - 1;

        // problem.AddResidualBlock(
        //     smoothness_cost_function,
        //     nullptr,
        //     &z(j),
        //     &z(j_right),
        //     &z(j_left),
        //     &z(j_top),
        //     &z(j_bottom)
        // );

        for (int i = 0; i < data.I.cols(); ++i) { // For each light
            int idx = j * data.I.cols() + i;

            ceres::CostFunction* photometric_cost_function =
                new ceres::AutoDiffCostFunction<PhotometricResidual, 1, 1, 1, 1>(
                    new PhotometricResidual(
                        data.I(j, i), rho(j),
                        data.Kinv,
                        data. Kinv * Eigen::Vector3d(x, y, 1.0),
                        data.light_positions[i]
                    )
                );

            // Add residual block with all 5 parameters
            problem.AddResidualBlock(
                photometric_cost_function   ,
                loss,
                &z(j),        // Current depth
                &z(j_right),  // Right neighbor
                &z(j_bottom)  // Bottom neighbor
            );

            double initial_depth = 2.147;
            problem.SetParameterLowerBound(&z(j), 0, initial_depth-0.05);
            problem.SetParameterUpperBound(&z(j), 0, initial_depth+0.05);
        }
    }

#pragma omp parallel for schedule(dynamic) num_threads(16)
    for (int j = 0; j < z.size(); ++j) {

        int x = j / image_height;
        int y = j % image_height;

        if (x == image_width - 1 && y == image_height - 1) continue;

        //if (x == 0 && y == 0) continue;
        //if (x == 0 && y == image_height - 1) continue;
        //if (x == image_width - 1 && y == 0) continue;
        //if (x == image_width - 1 && y == image_height -1) continue;

        if (x == 0 || x == image_width - 1 || y == 0 || y == image_height - 1) {
            problem.SetParameterBlockConstant(&z(j));
        }
    }


    // Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 1;
    options.max_linear_solver_iterations = 5;
    options.function_tolerance = 1e-9;
    options.gradient_tolerance = 1e-7;
    options.parameter_tolerance = 4e-9;
    options.jacobi_scaling = true;
    options.use_inner_iterations = false;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    options.initial_trust_region_radius = 1.0;  // default 10
    options.max_trust_region_radius = 10.0;    // Limit how large the trust region can grow

    options.num_threads = 16;

    // Create and add the callback for saving depth maps
    options.update_state_every_iteration = true;
    options.logging_type = ceres::PER_MINIMIZER_ITERATION;
    auto* callback = new DepthMapCallback(
        data.width, data.height, z.data(), iteration_count
    );
    options.callbacks.push_back(callback);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // std::cout << "Initial cost: " << summary.initial_cost << std::endl;
    // std::cout << "Final cost: " << summary.final_cost << std::endl;
    // std::cout << "Termination: " << summary.termination_type << std::endl;
    // std::cout << summary.FullReport() << "\n";
}

void optimizeAlbedo(Eigen::VectorXd &z, Eigen::VectorXd &rho, const PrecomputedData &data, int iteration_count) {

    ceres::Problem problem;
    int image_width = data.width;
    int image_height = data.height;


    auto loss = nullptr;
    // Add residuals for all pixels and lights
#pragma omp parallel for schedule(dynamic) num_threads(16)
    for (int j = 0; j < data.I.rows(); ++j) { // For each pixel
        int x = j / image_height; // column major (И shapes)
        int y = j % image_height;


        // Skip boundary pixels (no neighbors exist)
        if (x == image_width - 1 || y == image_height - 1) {
            continue;
        }

        // Neighbor indices (safe, since we skipped boundaries)
        int j_right = j + image_height;
        int j_bottom = j + 1;

        for (int i = 0; i < data.I.cols(); ++i) { // For each light
            int idx = j * data.I.cols() + i;

            ceres::CostFunction* albedo_cost_function =
                new ceres::AutoDiffCostFunction<AlbedoResidual, 1, 1>(
                    new AlbedoResidual(
                        data.I(j, i),
                        z(j), z(j_right), z(j_bottom),
                        data.Kinv,
                        data. Kinv * Eigen::Vector3d(x, y, 1.0),
                        data.light_positions[i]
                    )
                );

            // Add residual block with all 5 parameters
            problem.AddResidualBlock(
                albedo_cost_function,
                loss,
                &rho(j)        // Current albedo
            );

            problem.SetParameterUpperBound(&rho(j),0,10);
            problem.SetParameterLowerBound(&rho(j),0,0);
        }
    }


    // Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.preconditioner_type = ceres::IDENTITY;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 1;
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
    options.max_trust_region_radius = 20.0;

    options.num_threads = 16;

    // Create and add the callback for saving depth maps
    options.update_state_every_iteration = true;
    options.logging_type = ceres::PER_MINIMIZER_ITERATION;

    auto* callback = new AlbedoCallback(
        data.width, data.height, rho.data(), iteration_count
    );

    options.callbacks.push_back(callback);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

void optimizeDepthAndAlbedo(Eigen::VectorXd& z, Eigen::VectorXd& rho, const PrecomputedData& data) {

    std::cout << "Matrix sizes:" << std::endl;
    std::cout << "I: " << data.I.rows() << "x" << data.I.cols() << std::endl;
    std::cout << "light_positions size: " << data.light_positions.size() << std::endl;

    // Print value ranges
    std::cout << "rho range: " << rho.minCoeff() << " to " << rho.maxCoeff() << std::endl;

    // Print camera matrix
    std::cout << "Camera matrix K:\n" << data.K << std::endl;

    for (int i = 0; i < 50; ++i) {
        std::cout << "Depth iteration: " << i << std::endl;
        optimizeDepthMap(z, rho, data, i);
        std::cout << "Albedo iteration: " << i << std::endl;
        optimizeAlbedo(z, rho, data, i);
    }
}


