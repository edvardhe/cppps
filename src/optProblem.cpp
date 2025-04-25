#include "optProblem.h"
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <vector>
#include <c++/12/bits/std_thread.h>


// Implement constructor
DepthCostFunctor::DepthCostFunctor(int pixel_x, int pixel_y, int light_i,
    int width,
    double I_ji, double rho_j, double phi_i, 
    const Eigen::Matrix3d& K,
    const Eigen::Matrix3d& J,
    const double light_distance,
    const Eigen::Vector3d& light_dir,
    const double anisotropy)
: pixel_x(pixel_x), pixel_y(pixel_y), light_i(light_i), width(width),
K(K), J(J), I_ji(I_ji), rho_j(rho_j), phi_i(phi_i),
light_distance(light_distance), light_dir(light_dir),
anisotropy(anisotropy) {}

// Implement operator()
template <typename T>
bool DepthCostFunctor::operator()(const T* const z_j,
                                  const T* const z_right,
                                  const T* const z_left,
                                  const T* const z_bottom,
                                  const T* const z_top,
                                  T* residual) const {
    Eigen::Matrix<T, 3, 1> light_dir_t = light_dir.cast<T>();
    light_dir_t[2] *= T(-1.0); // Invert z-axis for depth
    T distance = T(light_distance);
    T anisotropy = T(anisotropy);

    // Calculate gradients using central differences
    T dz_x = (T(*z_right) - T(*z_left)) * T(0.5);
    T dz_y = (T(*z_bottom) - T(*z_top)) * T(0.5);
    T image_gradient = Eigen::Matrix<T,2,1>(dz_x, dz_y).norm();
    Eigen::Matrix<T, 3, 1> grad_z_neg1(dz_x, dz_y, T(-1.0));
    Eigen::Matrix<T, 3, 1> n = J.transpose() * grad_z_neg1;
    n.normalize();

    // Compute residual
    T psi = ceres::fmax(light_dir_t.dot(n) - T(1e-3), T(0.0));
    residual[0] = T(rho_j) * T(phi_i) * psi - T(I_ji);
    return true;
}


// Main function for depth optimization step
void optimizeDepthMap(Eigen::VectorXd& z, const PrecomputedData& data) {
    
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

        // Compute neighbor indices (these will be unique since we're skipping boundary pixels)
        int j_right = j + 1;
        int j_left = j - 1;
        int j_bottom = j + image_width;
        int j_top = j - image_width;

        // Handle boundary neighbors by using initial depth
        auto get_neighbor = [&](int neighbor_j) -> double {
            int nx = neighbor_j % image_width;
            int ny = neighbor_j / image_width;

            // If neighbor is out of bounds, return initial depth
            if (nx < 0 || nx >= image_width || ny < 0 || ny >= image_height)
                return initial_log_depth;

            return z(neighbor_j); // Use optimized value if available
        };

        // Create copies for boundary handling
        double z_right_val = get_neighbor(j_right);
        double z_left_val = get_neighbor(j_left);
        double z_bottom_val = get_neighbor(j_bottom);
        double z_top_val = get_neighbor(j_top);

        for (int i = 0; i < data.I.cols(); ++i) { // For each light
            if (data.weights(j, i) < 1e-3) continue;
            int idx = j * data.I.cols() + i;
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<DepthCostFunctor, 1, 1, 1, 1, 1, 1>(
                    new DepthCostFunctor(x, y, i,
                        data.width,
                        data.I(j, i), data.rho(j), data.phi(i),
                        data.K,
                        data.J_all_pixels[j],
                        data.light_distances[idx],
                        data.light_dirs[idx],
                        data.anisotropy[idx]
                    )
                );

            // Add residual block with all 5 parameters
            problem.AddResidualBlock(
                cost_function,
                new ceres::CauchyLoss(0.1),
                &z(j),        // Current depth
                // Use proxy variables for boundary neighbors
                x == image_width-1 ? &z_right_val : &z(j_right),
                x == 0 ? &z_left_val : &z(j_left),
                y == image_height-1 ? &z_bottom_val : &z(j_bottom),
                y == 0 ? &z_top_val : &z(j_top)
             );
        }
    }

    // After adding all residual blocks, set bounds for each z_j
    const double delta_meters = 0.05;  // Max deviation: 2 cm
    for (int j = 0; j < z.size(); ++j) {

        // Compute bounds in log space
        double initial_actual_depth = 2.147;
        double lower_actual = initial_actual_depth - delta_meters;
        double upper_actual = initial_actual_depth + delta_meters;

        // Ensure lower bound is positive
        if (lower_actual <= 0) lower_actual = 1e-6;

        // Convert to log-depth bounds
        double lower_log = log(lower_actual);
        double upper_log = log(upper_actual);

        // Apply constraints to z(j)
        problem.SetParameterLowerBound(&z(j), 0, lower_log);
        problem.SetParameterUpperBound(&z(j), 0, upper_log);
    }


    // Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 30;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-8;
    options.parameter_tolerance = 1e-8;

    // Use trust region method with dogleg strategy for better convergence
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.initial_trust_region_radius = 1e-1; // Start with moderate steps
    options.max_trust_region_radius = 1e2;      // Allow larger steps if needed
    options.min_trust_region_radius = 1e-6;     // Don't let steps get too small


    // For escaping local minima, sometimes a larger initial trust region helps
    options.initial_trust_region_radius = 1e1;

    // Multi-threading for speed
    options.num_threads = 16;

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    
    std::cout << "Initial cost: " << summary.initial_cost << std::endl;
    std::cout << "Final cost: " << summary.final_cost << std::endl;
    std::cout << "Termination: " << summary.termination_type << std::endl;
    
    // Check if optimization actually improved the solution
    if (summary.final_cost >= summary.initial_cost) {
        std::cout << "Warning: Optimization did not decrease cost!" << std::endl;
    }
}

