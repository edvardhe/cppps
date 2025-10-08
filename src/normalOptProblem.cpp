#include "normalOptProblem.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>

class NormalMapCallback : public ceres::IterationCallback {
public:
    NormalMapCallback(
        int width,
        int height,
        double* normal_data, // pointer to [nx, ny, nx, ny, ...]
        int iteration_count,
        const std::string& output_dir
    ) : width_(width),
        height_(height),
        normal_data_(normal_data),
        iteration_count_(iteration_count),
        output_dir_(output_dir) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
        // Create normal map matrix
        Eigen::MatrixXd normal_map(height_, width_ * 3); // [nx, ny, nz] per pixel

        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                int idx = (y * width_ + x) * 2;
                double nx = normal_data_[idx];
                double ny = normal_data_[idx + 1];
                double nz = std::sqrt(std::max(0.0, 1.0 - nx * nx - ny * ny));
                // Store as RGB
                normal_map(y, x * 3 + 0) = nx;
                normal_map(y, x * 3 + 1) = ny;
                normal_map(y, x * 3 + 2) = nz;
            }
        }

        // Convert to OpenCV Mat and normalize to [0,255]
        cv::Mat normal_image(height_, width_, CV_32FC3);
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                cv::Vec3f vec;
                vec[0] = static_cast<float>(normal_map(y, x * 3 + 0));
                vec[1] = static_cast<float>(normal_map(y, x * 3 + 1));
                vec[2] = static_cast<float>(normal_map(y, x * 3 + 2));
                normal_image.at<cv::Vec3f>(y, x) = vec;
            }
        }
        // Normalize to [0,255] for visualization
        cv::Mat normal_image_vis;
        normal_image = (normal_image + 1.0f) / 2.0f; // Map [-1,1] to [0,1]
        normal_image.convertTo(normal_image_vis, CV_8UC3, 255.0);

        std::filesystem::create_directories(output_dir_);
        std::string filename = output_dir_ + "/normal_iter_" + std::to_string(iteration_count_) + ".png";
        cv::imwrite(filename, normal_image_vis);

        return ceres::SOLVER_CONTINUE;
    }

private:
    int width_;
    int height_;
    double* normal_data_;
    int iteration_count_;
    std::string output_dir_;
};

struct NormalResidual {
    NormalResidual(
    const double I_ji, const double rho_j,
    const Eigen::Matrix3d& Kinv_t,
    const Eigen::Vector3d& ray,
    const double light_distance,
    const Eigen::Vector3d& light_direction,
    const Eigen::Vector3d& sphere_position,
    const double light_intensity
    )
: I_ji(I_ji), rho_j(rho_j), Kinv_t(Kinv_t), ray(ray),
  light_distance(light_distance), light_direction(light_direction), sphere_position(sphere_position),
  light_intensity(light_intensity) {}

    template <typename T>
    bool operator()(const T* const x_j,
                    const T* const y_j,
                    T* residual) const {
        // --- Gradient Calculation with Boundary Handling ---

        Eigen::Matrix<T, 3, 1> n = Eigen::Matrix<T, 3, 1>(T(x_j[0]), T(y_j[0]), T(1.0));
        n.normalize();

        Eigen::Matrix<T, 3, 1> light_pos = light_direction.cast<T>() * light_distance + sphere_position.cast<T>();
        Eigen::Matrix<T, 3, 1> light_to_point = light_pos - x_j_3D;
        T distance = T(light_to_point.norm());
        Eigen::Matrix<T, 3, 1> s = light_to_point / (distance * distance * distance);

        T incoming_light = T(s.transpose() * n);

        T light_estimate = T(ceres::fmax(incoming_light,T(0.0)));
        T albedo_adjusted_estimate = light_estimate * T(rho_j) * light_intensity;

        // Compute residual
        residual[0] = (T(I_ji) - albedo_adjusted_estimate);
        return true;
    }
private:
    double I_ji;
    double rho_j;
    Eigen::Matrix3d Kinv_t;
    Eigen::Vector3d ray;
    double light_intensity;
    double light_distance;
    Eigen::Vector3d light_direction;
    Eigen::Vector3d sphere_position;
};

struct AlbedoResidual {
    AlbedoResidual(
    const double I_ji,
    const double z_j,
    const double z_right,
    const double z_bottom,
    const double light_distance,
    const Eigen::Matrix3d& Kinv_t,
    const Eigen::Vector3d ray,
    const Eigen::Vector3d light_direction,
    const Eigen::Vector3d sphere_position,
    const double light_intensity)
    : I_ji(I_ji), z_j(z_j), z_right(z_right), z_bottom(z_bottom), light_distance(light_distance), Kinv_t(Kinv_t),
      ray(ray), light_direction(light_direction), sphere_position(sphere_position), light_intensity(light_intensity) {}

    template <typename T>
    bool operator()(const T* const rho_j,
                    T* residual) const {
        // --- Gradient Calculation with Boundary Handling ---
        T dz_x, dz_y;

        // forward difference
        dz_x = T(z_right - z_j);
        dz_y = T(z_bottom - z_j);

        Eigen::Matrix<T, 3, 1> grad_z_neg1(dz_x, dz_y, T(-1.0));
        Eigen::Matrix<T, 3, 1>  x_j_3D = T(z_j) * ray.cast<T>();

        Eigen::Matrix<T, 3, 1> light_pos = light_direction.cast<T>() * T(light_distance) + sphere_position.cast<T>();
        Eigen::Matrix<T, 3, 1> light_to_point = light_pos - x_j_3D;

        T distance = T(light_to_point.norm());
        Eigen::Matrix<T, 3, 1> s = light_to_point / (distance * distance * distance);

        T incoming_light = T(s.transpose() * (Kinv_t * grad_z_neg1));

        T light_estimate = ceres::fmax(incoming_light,T(0.0));
        T albedo_adjusted_estimate = T(light_estimate * (*rho_j)) * T(light_intensity);

        // Compute residual
        residual[0] = (T(I_ji) - albedo_adjusted_estimate);
        return true;
    }
private:
    double I_ji;
    double z_j, z_right, z_bottom;
    double light_distance;
    double light_intensity;
    Eigen::Matrix3d Kinv_t;
    Eigen::Vector3d ray;
    Eigen::Vector3d light_direction;
    Eigen::Vector3d sphere_position;
};

void optimizeNormalMap(Eigen::VectorXd& n, const Eigen::VectorXd& rho, const Eigen::VectorXd& light_distances, const PrecomputedData& data) {
    int n_pixels = data.width * data.height;

    // Ceres problem setup
    ceres::Problem problem;

    int image_width = data.width;
    int image_height = data.height;

    for (int j = 0; j < data.I.rows(); ++j) { // For each pixel
        int x = j / image_height; // column major (И shapes)
        int y = j % image_height;

        int glob_x = data.start_x + x;
        int glob_y = data.start_y + y;

        if (x == image_width - 1 || y == image_height - 1) {
            continue;
        }


        for (int i = 0; i < data.I.cols(); ++i) { // For each light
            int idx = j * data.I.cols() + i;
            ceres::CostFunction* normal_cost_function =
                new ceres::AutoDiffCostFunction<NormalResidual, 1, 1, 1>(
                    new NormalResidual(
                        data.I(j, i),
                        rho(j),
                        data.Kinv_t,
                        data.Kinv * Eigen::Vector3d(glob_x, glob_y, 1.0),
                        light_distances[i],
                        data.light_dirs[i],
                        data.sphere_position,
                        1.0
                    )
                );

            problem.AddResidualBlock(normal_cost_function, nullptr, &n(2 * j), &n(2 * j + 1));
        }

    /// Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 5;
    options.max_linear_solver_iterations = 5;
    options.function_tolerance = 1e-9;
    options.gradient_tolerance = 1e-7;
    options.parameter_tolerance = 4e-9;
    options.use_nonmonotonic_steps = true;
    options.jacobi_scaling = true;
    options.use_inner_iterations = false;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    options.initial_trust_region_radius = 100.0;  // default 10
    //options.max_trust_region_radius = 10.0;    // Limit how large the trust region can grow

    options.num_threads = 16;

    // Create and add the callback for saving depth maps
    options.update_state_every_iteration = true;
    options.logging_type = ceres::PER_MINIMIZER_ITERATION;
    auto* callback = new NormalMapCallback(
        data.width, data.height, n.data(), 1, std::string(PROJECT_DIR) + "/NormalIterations"
    );
    options.callbacks.push_back(callback);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    options.callbacks.clear();

    std::cout << summary.FullReport() << std::endl;
}