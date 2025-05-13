#include <iostream>
#include <vector>
#include <filesystem>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include "depthMapHandler.h"
#include "MatrixProcessing.h"
#include "optProblem.h"

namespace fs = std::filesystem;

using json = nlohmann::json;


std::vector<ROI> getRois(int image_width, int image_height, int roi_width, int roi_height) {
    std::vector<ROI> rois;

    // Define overlap size
    const int overlap = 10;

    // Calculate effective step size (accounting for overlap)
    int step_width = roi_width - overlap;
    int step_height = roi_height - overlap;

    // Calculate number of ROIs in each dimension
    int num_rois_x = std::ceil(static_cast<float>(image_width - overlap) / step_width);
    int num_rois_y = std::ceil(static_cast<float>(image_height - overlap) / step_height);

    // Generate overlapping ROIs
    for (int y = 0; y < num_rois_y; ++y) {
        for (int x = 0; x < num_rois_x; ++x) {
            ROI roi;

            // Store grid position
            roi.x = x;
            roi.y = y;

            // Calculate starting position
            roi.start_x = x * step_width;
            roi.start_y = y * step_height;

            // Handle edge case: ensure we don't exceed image dimensions
            if (roi.start_x + roi_width > image_width) {
                roi.start_x = image_width - roi_width;
            }

            if (roi.start_y + roi_height > image_height) {
                roi.start_y = image_height - roi_height;
            }

            // Set ROI dimensions
            roi.width = roi_width;
            roi.height = roi_height;

            rois.push_back(roi);
        }
    }

    return rois;
}

Eigen::MatrixXd runWithRoi(Eigen::Matrix3d K_pixel, int start_x, int start_y, int roi_width, int roi_height) {
    // Load images with ROI
    std::string path = "/home/edvard/dev/projects/cppPS/ratioImages";
    std::vector<std::string> image_names;
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

    // Each column is one image, each row is one pixels intensity across images

    // 1. Initialize data structures
    PrecomputedData data;

    // 2. Populate precomputed data
    data.start_x = start_x;   // X starting position
    data.start_y = start_y;   // Y starting position
    data.width = roi_width;   // Image width
    data.height = roi_height; // Image height
    data.K = K_pixel;         // Camera intrinsics
    data.I = images;          // Input images
    data.rho = false ? Eigen::VectorXd(n_pixels).setConstant(1.0) : images.rowwise().mean(); // Mean per pixel
    data.light_positions = light_positions; // Your calibration data

    // 3. Compute Jacobians for all pixels
    precomputeJacobian(data);

    std::vector<Eigen::Vector3d> precomputed_light_dirs; // [pixel][light]
    std::vector<double> precomputed_distances;           // [pixel][light]

    double initial_depth = 2.147;
    precomputeLightVectors(data, initial_depth);


    // 4. Initialize depth
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::VectorXd z(n_pixels);
    z.setConstant(initial_depth);  // Meter initialization

    // z-step: Update depth map
    std::cout << "I rows: " << data.I.rows() << ", cols: " << data.I.cols() << std::endl;
    optimizeDepthMap(z, z.data(), data);

    Eigen::MatrixXd depth_map(Eigen::Map<Eigen::MatrixXd>(z.data(), data.height, data.width));

    return depth_map;
}

int main(int argc, char** argv) {

    Eigen::initParallel();
    Eigen::setNbThreads(4);

    int image_width = 4460;
    int image_height = 8736;

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

    K_pixel << f * px_per_m_x, 0.0,             image_width / 2.0,
               0.0,             f * px_per_m_y, image_height / 2.0,
               0.0,             0.0,            1.0;


    // In main(), replace the image loading line with:
    int start_x = 3418;  // Your desired X starting position
    int start_y = 4832;  // Your desired Y starting position
    int roi_width = 569;  // Your desired width
    int roi_height = 243; // Your desired height

    std::vector<ROI> regions_of_interest = getRois(image_width, image_height, roi_width, roi_height);

    Eigen::MatrixXd z = runWithRoi(K_pixel, start_x, start_y, roi_width, roi_height);
    return 0;

    for (const ROI& roi : regions_of_interest) {
        std::string patch_dir = "/home/edvard/dev/projects/cppPS/depthPatches";
        std::string patch_path = patch_dir + "/depth_" + std::to_string(roi.x) + "_" + std::to_string(roi.y) + ".dmap";
        if (fs::exists(patch_path)) {
            continue;
        }
        std::cout << "ROI at grid position (" << roi.x << ", " << roi.y
                  << ") starts at (" << roi.start_x << ", " << roi.start_y
                  << ") with size " << roi.width << "x" << roi.height << std::endl;


        Eigen::MatrixXd z = runWithRoi(K_pixel, roi.start_x, roi.start_y, roi.width, roi.height);
        depth::saveDepthMap(z, "/home/edvard/dev/projects/cppPS/depthMapPatches",
                            std::to_string(roi.x) + "_" + std::to_string(roi.y));
        depth::saveDepthMapBinary(z.data(), roi, patch_dir);
    }

    // After generating and saving all the depth map patches, stitch them together
    std::string patch_dir = "/home/edvard/dev/projects/cppPS/depthPatches";
    Eigen::MatrixXd mega_depth_map = depth::stitchDepthMaps(patch_dir);
    depth::saveDepthMap(mega_depth_map, "/home/edvard/dev/projects/cppPS/megaDepthMap", "mega");

}

