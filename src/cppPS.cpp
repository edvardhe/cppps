#include <iostream>
#include <vector>
#include <filesystem>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include "MatrixProcessing.h"
#include "optProblem.h"

namespace fs = std::filesystem;

using json = nlohmann::json;

int main(int argc, char** argv) {

    Eigen::initParallel();
    Eigen::setNbThreads(16);

    float f = 4.91 / 100.0;  // m

    float sensor_width = 4.38 / 100.0;  // m
    float sensor_height = 3.29 / 100.0;  // m

    float sensor_width_pixels = 11648.0; // pixels
    float sensor_height_pixels = 8736.0; // pixels

    float px_per_m_x = sensor_width_pixels / sensor_width;
    float px_per_m_y = sensor_height_pixels / sensor_height;

    Eigen::Matrix3d K, K_pixel, K_test;
    K << f, 0.0, sensor_width / 2.0,
         0.0, f, sensor_height / 2.0,
         0.0, 0.0, 1.0;

    K_pixel << f * px_per_m_x, 0.0,             sensor_width_pixels / 2.0,
               0.0,             f * px_per_m_y, sensor_height_pixels / 2.0,
               0.0,             0.0,            1.0;

    //if (argc != 2) {
    //    std::cerr << "Usage: " << argv[0] << " <image_directory>" << std::endl;
    //    return 1;
    //}

    // 1. Load images from directory
    std::string path = "/home/edvard/dev/projects/cppPS/ratioImages";
    std::vector<std::string> image_names;

    // In main(), replace the image loading line with:
    int start_x = 5600;  // Your desired X starting position
    int start_y = 3446;  // Your desired Y starting position
    int roi_width = 982;  // Your desired width
    int roi_height = 544; // Your desired height

    // Load images with ROI
    Eigen::MatrixXd images = loadImagesToObservationMatrix(path, image_names,
                                                      start_x, start_y, 
                                                      roi_width, roi_height);

    // Load images
    //Eigen::MatrixXd images = loadImagesToObservationMatrix(path, image_names, image_width, image_height);
    int n_pixels = images.rows();
    int n_images = images.cols();

    std::vector<Eigen::Vector3d> light_positions = loadJsonToPositionMatrix("/home/edvard/dev/projects/cppPS/light_positions.json", image_names);

    // Print stats
    std::cout << "Loaded " << n_images << " images with " << n_pixels 
                << " pixels each" << std::endl;

    // Now 'I' can be used in Algorithm 2
    // Each column is one image, each row is one pixel's intensity across images

    // 1. Initialize data structures
    PrecomputedData data;

    // 2. Populate precomputed data
    data.start_x = start_x; // X starting position
    data.start_y = start_y; // Y starting position
    data.width = roi_width; // Image width
    data.height = roi_height; // Image height
    data.K = K_pixel; // Camera intrinsics
    data.I = images; // Input images
    data.rho = true ? Eigen::VectorXd(n_pixels).setConstant(1.0) : images.rowwise().mean(); // Mean per pixel
    data.phi = true ? Eigen::VectorXd(n_images).setConstant(1.0) : images.colwise().mean(); // Mean per light
    data.light_positions = light_positions; // Your calibration data
    data.weights = Eigen::MatrixXd::Ones(data.I.rows(), data.I.cols());

    // Initialize LED parameters for each light
    data.mu_i.resize(data.I.cols(), 1.0);  // Default anisotropy parameter
    data.n_s_i.resize(data.I.cols(), Eigen::Vector3d(0, 0, 1));  // Default LED direction

    // 3. Compute Jacobians for all pixels
    precomputeJacobian(data);

    std::vector<Eigen::Vector3d> precomputed_light_dirs; // [pixel][light]
    std::vector<double> precomputed_distances;           // [pixel][light]

    double initial_depth = 2.147;
    precomputeLightVectors(data, initial_depth);
    

    // 4. Initialize depth (log-depth)
    // Initialize depth as log-depth (z = log(̃z))
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::VectorXd z(n_pixels);
    //z.setConstant(std::log(initial_depth)); // Log-depth initialization
    z.setConstant(initial_depth);             // Meter initialization
    //Add small noise to log-depth
    //z = z.unaryExpr([](double x) { return x + 0.0001 * (rand() / double(RAND_MAX) - 0.0001); });

    // z-step: Update depth map
    std::cout << "I rows: " << data.I.rows() << ", cols: " << data.I.cols() << std::endl;
    optimizeDepthMap(z, z.data(), data);
    //optim
    //runFullOptimization(data);
    // Optional: Check convergence
    // if (checkConvergence(z)) break;
    Eigen::VectorXd depth = z.array().exp(); // Convert to actual depth

        // After solving, normalize and save depth map
        double z_min = depth.minCoeff();
        double z_max = depth.maxCoeff();
        
        std::cout << "Depth range: " << z_min << " to " << z_max << std::endl;
        
        // Create OpenCV Mat for visualization
        cv::Mat depth_map(roi_height, roi_width, CV_8UC1);
        
        // Normalize to [0,255] for visualization
        for (int j = 0; j < z.size(); ++j) {
            int x = j % roi_width;
            int y = j / roi_width;
            double normalized_depth = (depth(j) - z_min) / (z_max - z_min);
            depth_map.at<uchar>(y, x) = static_cast<uchar>(normalized_depth * 255);
        }
        // Save the depth map
        cv::imwrite("../depth_map.png", depth_map);

    //}

    // Final depth map: z contains log-depth values
    //Eigen::VectorXd depth = z.array().exp(); // Convert to actual depth
    return 0;
}