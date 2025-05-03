#include "optProblem.h"

#include <filesystem>
#include <fstream>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <ceres/manifold.h>
#include <utility>
#include <vector>
#include <c++/12/bits/std_thread.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

/**
 * Callback for saving depth map images during optimization
 */
class DepthMapCallback : public ceres::IterationCallback {
public:
    DepthMapCallback(
        int width,
        int height,
        double* depth_data,
        size_t depth_size,
        std::string output_dir = "/home/edvard/dev/projects/cppPS/forReport/depthMapIterations"
    ) : width_(width),
        height_(height),
        depth_data_(depth_data),
        depth_size_(depth_size),
        output_dir_(std::move(output_dir)),
        iteration_count_(0) {

        // Create output directory if it doesn't exist
        if (!std::filesystem::exists(output_dir_)) {
            std::filesystem::create_directories(output_dir_ + "/gray");
            std::filesystem::create_directories(output_dir_ + "/cool");
        }
    }

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
        // Save the current depth map
        saveDepthMap();

        saveDepthMapAsObj();
        // Increment iteration counter
        iteration_count_++;

        // Continue optimization
        return ceres::SOLVER_CONTINUE;
    }

private:
    void saveDepthMap() {
        // Find min/max depth for normalization
        double min_depth = std::numeric_limits<double>::max();
        double max_depth = std::numeric_limits<double>::lowest();

        Eigen::Map<Eigen::VectorXd> depth_map_(depth_data_, depth_size_);
        //auto depth_array = depth_map_.array().exp();
        auto depth_array = depth_map_.array();

        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                int idx = y * width_ + x;
                min_depth = std::min(min_depth, depth_array(idx));
                max_depth = std::max(max_depth, depth_array(idx));
            }
        }

        // Create normalized depth map
        cv::Mat depth_map(height_, width_, CV_32FC1, cv::Scalar(0)); // Initialize with 0

        // Only process non-edge pixels
        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                int idx = y * width_ + x;
                auto normalized_depth = static_cast<float>(
                    (depth_array(idx) - min_depth) / (max_depth - min_depth)
                );
                depth_map.at<float>(y, x) = normalized_depth;
            }
        }

        // Create output image
        cv::Mat output;
        depth_map.convertTo(output, CV_8UC1, 255.0);

        // Save the image
        std::string filename = output_dir_ + "/gray/depth_map_iter_" +
                              std::to_string(iteration_count_) + ".png";
        cv::imwrite(filename, output);

        // Apply colormap if requested
        cv::applyColorMap(output, output, cv::COLORMAP_MAGMA);

        filename = output_dir_ + "/cool/depth_map_iter_" +
                   std::to_string(iteration_count_) + ".png";
        cv::imwrite(filename, output);
    }
    /**
     * Saves a depth map as a 3D mesh in Wavefront .obj format
     *
     * @param width Width of the depth map
     * @param height Height of the depth map
     * @param depth_data Pointer to depth map data (flattened 2D array)
     * @param depth_size Size of the depth data array
     * @param filename Path to the output .obj file
     * @param scale_x X-axis scale factor (default: 1.0)
     * @param scale_y Y-axis scale factor (default: 1.0)
     * @param scale_z Z-axis scale factor (default: 1.0)
     * @return True if successful, false otherwise
     */
    bool saveDepthMapAsObj()
    {
        int width = width_;
        int height = height_;

        const double* depth_data = depth_data_;

        size_t depth_size = depth_size_;

        double scale_x = 1.0;
        double scale_y = 1.0;
        double scale_z = -3000.0;

        if (depth_size != width * height) {
            std::cerr << "Error: depth_size doesn't match width*height" << std::endl;
            return false;
        }

        std::string filename = output_dir_ + "/obj/depth_obj_iter_" + std::to_string(iteration_count_) + ".obj";

        std::ofstream objFile(filename);
        if (!objFile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return false;
        }

        // Write header
        objFile << "# Depth map exported as OBJ\n";
        objFile << "# Width: " << width << ", Height: " << height << "\n";

        // Map depth data
        Eigen::Map<const Eigen::VectorXd> depth_map(depth_data, depth_size);
        auto depth_array = depth_map.array();

        // Find min/max depth for normalization
        double min_depth = depth_array.minCoeff();
        double max_depth = depth_array.maxCoeff();
        std::cout << "Depth range: " << min_depth << " to " << max_depth << std::endl;

        // Write vertices
        // The coordinate system is: X right, Y up, Z backward (toward viewer)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                double depth = depth_array(idx);

                // Normalize x and y to [-1, 1] range
                double norm_x = (2.0 * x / (width - 1) - 1.0) * scale_x;
                double norm_y = (1.0 - 2.0 * y / (height - 1)) * scale_y; // Flip Y to match 3D convention

                objFile << "v " << norm_x << " " << norm_y << " "
                       << depth * scale_z << "\n";
            }
        }

        // Write faces (triangles)
        // Use counter-clockwise winding order
        for (int y = 0; y < height - 1; y++) {
            for (int x = 0; x < width - 1; x++) {
                // 0-based indices of the four corners of the current grid cell
                int idx00 = y * width + x + 1;            // +1 because OBJ indices start at 1
                int idx01 = (y + 1) * width + x + 1;
                int idx10 = y * width + (x + 1) + 1;
                int idx11 = (y + 1) * width + (x + 1) + 1;

                // Write two triangles for this grid cell
                objFile << "f " << idx00 << " " << idx10 << " " << idx11 << "\n";
                objFile << "f " << idx00 << " " << idx11 << " " << idx01 << "\n";
            }
        }

        objFile.close();
        //std::cout << "Depth map saved as OBJ file: " << filename << std::endl;
        return true;
    }

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

class Rank1Manifold : public ceres::Manifold {
public:
    Rank1Manifold(int num_pixels, int num_lights)
    : num_pixels_(num_pixels), num_lights_(num_lights) {}

    // Ambient dimension (size of parameter vector: ρ + ϕ)
    int AmbientSize() const override {
        return num_pixels_ + num_lights_;
    }

    // Tangent space dimension (accounts for scale ambiguity)
    int TangentSize() const override {
        return num_pixels_ + num_lights_ - 1;
    }

    // Move parameters along the tangent vector `delta`
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
        std::copy(x, x + AmbientSize(), x_plus_delta);
        for (int i = 0; i < TangentSize(); ++i) {
            x_plus_delta[i] += delta[i];
        }
        return true;
    }

    // Compute the tangent vector from `x` to `y`
    bool Minus(const double* y, const double* x, double* delta) const override {
        for (int i = 0; i < TangentSize(); ++i) {
            delta[i] = y[i] - x[i];
        }
        return true;
    }

    // Jacobian of Plus (identity for simplicity)
    bool PlusJacobian(const double* x, double* jacobian) const override {
        Eigen::Map<Eigen::MatrixXd>(jacobian, AmbientSize(), TangentSize())
            .setIdentity();
        return true;
    }

    // Jacobian of Minus (identity for simplicity)
    bool MinusJacobian(const double* x, double* jacobian) const override {
        Eigen::Map<Eigen::MatrixXd>(jacobian, TangentSize(), AmbientSize())
            .setIdentity();
        return true;
    }
private:
    int num_pixels_, num_lights_;
};

struct AlbIntensityResidual {
    const double I_ji;   // Observed intensity
    const double Psi_ji; // Precomputed geometric term (from depth z)
    const double weight; // Robust weight (Cauchy, etc.)

    template <typename T>
    bool operator()(const T* const rho_j, const T* const phi_i, T* residual) const {
        T theta_ji = (*rho_j) * (*phi_i); // θ[j,i] = ρ[j] * ϕ[i]
        T predicted = theta_ji * T(Psi_ji);
        residual[0] = T(weight) * (predicted - T(I_ji));
        return true;
    }
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

// Implement operator()
template <typename T>
bool DepthCostFunctor::operator()(
    const T* const z_j,
    const T* const z_right,
    const T* const z_bottom,
    T* residual) const {
        // TODO :: FIXA LJUSET!! fixa också grannarna så att de använder de grannar som finns det fuckar med kanterna

        // --- Gradient Calculation with Boundary Handling ---
        T dz_x, dz_y;

        //// center difference
        //dz_x = (*z_right - *z_left) / T(2.0);
        //dz_y = (*z_bottom - *z_top) / T(2.0);

        //// forward difference
        dz_x = (*z_j - *z_right);
        dz_y = (*z_j - *z_bottom);

        Eigen::Matrix<T, 3, 1> grad_z_neg1(dz_x, dz_y, T(-1.0));
        Eigen::Matrix<T, 3, 1> n = J.transpose() * grad_z_neg1;
        n.normalize();

        // Roughness penalty
        T curvature = dz_x*dz_x + dz_y*dz_y;
        T roughness_penalty_factor = T(7e6) * curvature;

        // Depth prior
        T depth_prior = T(2.147);  // Encourage depths > 2.0 units
        T penalty = ceres::fmax(T(0.0), depth_prior - *z_j);
        T depth_prior_penalty = T(5000.0) * penalty;  // Add as a second residual

        // Compute lighting
        Eigen::Matrix<T, 3, 1> light_dir_t = light_dir.cast<T>();
        T distance = T(light_distance);
        T anisotropy = T(anisotropy);

        T falloff = T(1.0) / (distance * distance);
        Eigen::Matrix<T, 3, 1> s = falloff * light_dir_t;
        Eigen::Matrix<T, 1, 3> JsT = (J * s).transpose();
        T incoming_light = JsT * grad_z_neg1;
        T light_estimate = ceres::fmax(s.dot(n),T(0.0));
        T albedo_adjusted_estimate = T(rho_j) * T(phi_i) * light_estimate;

        // Compute residual
        residual[0] = abs(albedo_adjusted_estimate - T(I_ji));
        residual[1] = roughness_penalty_factor;
        residual[2] = depth_prior_penalty;
        return true;
}

// Main function for depth optimization step
void optimizeDepthMap(Eigen::VectorXd& z, double* z_p, const PrecomputedData& data) {
    
    // Add debug prints at the start of the function
    std::cout << "Matrix sizes:" << std::endl;
    std::cout << "I: " << data.I.rows() << "x" << data.I.cols() << std::endl;
    std::cout << "weights: " << data.weights.rows() << "x" << data.weights.cols() << std::endl;
    std::cout << "n_s_i size: " << data.n_s_i.size() << std::endl;
    std::cout << "mu_i size: " << data.mu_i.size() << std::endl;
    std::cout << "light_positions size: " << data.light_positions.size() << std::endl;

    // Print value ranges
    std::cout << "Value ranges:" << std::endl;
    std::cout << "I range: " << data.I.minCoeff() << " to " << data.I.maxCoeff() << std::endl;
    std::cout << "rho range: " << data.rho.minCoeff() << " to " << data.rho.maxCoeff() << std::endl;
    std::cout << "phi range: " << data.phi.minCoeff() << " to " << data.phi.maxCoeff() << std::endl;
    
    // Print a few light positions
    std::cout << "First light position: " << data.light_positions[0].transpose() << std::endl;
    
    // Print camera matrix
    std::cout << "Camera matrix K:\n" << data.K << std::endl;

    ceres::Problem problem;
    int image_width = data.width;
    int image_height = data.height;

    constexpr double initial_depth = 2.147; // Your initial depth value
    const double initial_log_depth = log(initial_depth);

    // Add residuals for all pixels and lights
    for (int j = 0; j < data.I.rows(); ++j) { // For each pixel
        int x = j % image_width;
        int y = j / image_width;

        // Skip boundary pixels (no neighbors exist)
        if (x == image_width - 1 || y == image_height - 1) {
            continue;
        }

        // Neighbor indices (safe, since we skipped boundaries)
        int j_right = j + 1;
        int j_bottom = j + image_width;

        for (int i = 0; i < data.I.cols(); ++i) { // For each light
            if (data.weights(j, i) < 1e-3) continue;
            int idx = j * data.I.cols() + i;
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<DepthCostFunctor, 3, 1, 1, 1>(
                    new DepthCostFunctor(
                        data.I(j, i), data.rho(j), data.phi(i),
                        data.J_all_pixels[j],
                        data.light_distances[idx],
                        data.light_dirs[idx],
                        data.anisotropy[idx]
                    )
                );

            // Add residual block with all 5 parameters
            problem.AddResidualBlock(
                cost_function,
                new ceres::CauchyLoss(1),
                &z(j),        // Current depth
                &z(j_right),  // Right neighbor
                &z(j_bottom)  // Bottom neighbor
            );
        }
    }

    // After adding all residual blocks, set bounds for each z_j
    const double delta_meters = 0.03;
    // Compute bounds in log space
    double initial_actual_depth = 2.147;
    double lower_actual = initial_actual_depth - delta_meters;
    double upper_actual = initial_actual_depth + delta_meters;

    for (int j = 0; j < z.size(); ++j) {

        int x = j % image_width;
        int y = j / image_width;

        if (x == 0 || x == image_width - 1 || y == 0 || y == image_height - 1) {
            if (x == image_width - 1 && y == image_height - 1) continue;
            problem.SetParameterBlockConstant(&z(j));
            // OR: Set to a neighbor's value
            // if (x == image_width - 1) z(j) = z(j - 1);
        }
    }


    // Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::CGNR;
    options.preconditioner_type = ceres::IDENTITY;
    options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-8;
    options.parameter_tolerance = 1e-8;

    // Switch to line search
    options.minimizer_type = ceres::LINE_SEARCH;

    // Line search specific options
    options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;  // or ceres::STEEPEST_DESCENT, ceres::NONLINEAR_CONJUGATE_GRADIENT
    options.line_search_type = ceres::WOLFE;           // or ceres::ARMIJO
    options.line_search_interpolation_type = ceres::CUBIC;

    // Line search parameters
    options.max_lbfgs_rank = 20;  // For LBFGS direction
    options.use_approximate_eigenvalue_bfgs_scaling = true;
    options.line_search_sufficient_function_decrease = 1e-4;
    options.line_search_sufficient_curvature_decrease = 0.9;
    options.max_line_search_step_contraction = 1e-3;
    options.min_line_search_step_contraction = 0.6;
    options.max_num_line_search_step_size_iterations = 20;
    options.max_num_line_search_direction_restarts = 5;
    options.line_search_sufficient_curvature_decrease = 0.9;

    // Multi-threading for speed
    options.num_threads = 16;
    // Create and add the callback for saving depth maps
    options.update_state_every_iteration = true;
    options.logging_type = ceres::PER_MINIMIZER_ITERATION;
    auto* callback = new DepthMapCallback(
        data.width, data.height, z_p, z.size()
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

Eigen::MatrixXd computePsi(Eigen::VectorXd::Nested matrix, const PrecomputedData & data){
    return matrix;
}

void optimizeAlbedoAndIntensities(
    Eigen::VectorXd& rho,        // Albedo (n pixels)
    Eigen::VectorXd& phi,        // Light intensities (m lights)
    const Eigen::VectorXd& z,    // Current depth estimate
    const PrecomputedData& data  // Contains I, J, light positions, etc.
) {
    ceres::Problem problem;
    int num_pixels = data.width * data.height;
    int num_lights = data.light_positions.size();

    // Set up manifold for rank-1 constraint
    problem.AddParameterBlock(rho.data(), num_pixels);
    problem.AddParameterBlock(phi.data(), num_lights);

    Rank1Manifold* manifold = new Rank1Manifold(num_pixels, num_lights);
    problem.SetManifold(rho.data(), manifold);
    problem.SetManifold(phi.data(), manifold);

    // Precompute Ψ_ji (geometric terms) using current depth z
    Eigen::MatrixXd Psi = computePsi(z, data); // Implement this function

    // Add residuals for all pixels/lights
    for (int j = 0; j < num_pixels; ++j) {
        for (int i = 0; i < num_lights; ++i) {
            if (data.weights(j, i) < 1e-3) continue; // Skip invalid/shadowed

            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<AlbIntensityResidual, 1, 1, 1>(
                    new AlbIntensityResidual{
                        data.I(j, i),
                        Psi(j, i),
                        data.weights(j, i)
                    }
                );

            problem.AddResidualBlock(
                cost_function,
                new ceres::CauchyLoss(1), // Robust loss
                &rho(j),  // Albedo parameter for pixel j
                &phi(i)   // Intensity parameter for light i
            );
        }
    }

    // Solve
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

void runFullOptimization(PrecomputedData& data) {
    // Initialize depth, albedo, and intensities
    Eigen::VectorXd z = Eigen::VectorXd::Constant(data.width * data.height, 2.147);
    data.rho = Eigen::VectorXd::Ones(data.width * data.height); // Initial albedo = 1
    data.phi = Eigen::VectorXd::Ones(data.I.cols());            // Initial intensities = 1

    // Alternating optimization loop
    for (int iter = 0; iter < 10; ++iter) {
        // Step 1: Optimize depth using current rho/phi
        optimizeDepthMap(z, z.data(), data);

        // Step 2: Optimize albedo/intensities using current depth
        optimizeAlbedoAndIntensities(data.rho, data.phi, z, data);

        // Check convergence (e.g., cost change < threshold)
        //if (converged) break;
    }
}

