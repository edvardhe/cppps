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

            roi.name = "ROI_" + std::to_string(x) + "_" + std::to_string(y);

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

Eigen::MatrixXd runWithRoi(Eigen::Matrix3d K_pixel, ROI roi) {
    // Load images with ROI
    std::cout << "Computing region of interest: " + roi.name << std::endl;
    std::string path = "/home/edvard/dev/projects/cppPS/color";
    std::vector<std::string> image_names;
    Eigen::MatrixXd images = loadImagesToObservationMatrix(path, image_names,
                                                      roi.start_x, roi.start_y,
                                                      roi.width, roi.height);

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
    data.start_x = roi.start_x;   // X starting position
    data.start_y = roi.start_y;   // Y starting position
    data.width = roi.width;   // Image width
    data.height = roi.height; // Image height
    data.K_pixel = K_pixel;
    data.K = data.K_pixel;         // Camera intrinsics
    data.Kinv = data.K.inverse();
    data.Kinv_t = data.K.inverse().transpose();
    data.I = images;          // Input images
    data.light_positions = light_positions; // Your calibration data

    //precomputeJacobian(data);

    double initial_depth = 2.147;
    //precomputeLightVectors(data, initial_depth);
    //precomputeGeometricTerms(data);


    // Initialize depth
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::VectorXd z(n_pixels);
    z.setConstant(initial_depth);  // Meter initialization

    double perturbation_scale = 0.005;

    //std::srand(std::time(nullptr));
    //for (int j = 0; j < z.size(); ++j) {
    //    double random_value = 2.0 * (static_cast<double>(std::rand()) / RAND_MAX) - 1.0;
    //    z(j) += random_value * perturbation_scale;
    //}

    // Initialize albedo
    Eigen::VectorXd rho;
    rho = false ? Eigen::VectorXd(n_pixels).setConstant(1.0) : images.rowwise().mean(); // Mean per pixel

    // z-step: Update depth map
    std::cout << "I rows: " << data.I.rows() << ", cols: " << data.I.cols() << std::endl;

    optimizeDepthAndAlbedo(z, rho, data);

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

    double px_per_m = (px_per_m_x + px_per_m_y) / 2.0;

    K_pixel << f * px_per_m_x, 0.0,             image_width / 2.0,
               0.0,             f * px_per_m_y, image_height / 2.0,
               0.0,             0.0,            1.0;


    // Small Letters Example
    ROI small_letters_roi;
    small_letters_roi.name = "small_letters";
    small_letters_roi.start_x = 3418;
    small_letters_roi.start_y = 4850;
    small_letters_roi.width = 569;
    small_letters_roi.height = 223;

    // R example
    ROI R_roi;
    R_roi.name = "R";
    R_roi.start_x = 2061;
    R_roi.start_y = 3514;
    R_roi.width = 665;
    R_roi.height = 371;

    // Really small R
    ROI really_small_R_roi;
    really_small_R_roi.name = "small_R";
    really_small_R_roi.start_x = 2061;
    really_small_R_roi.start_y = 3514;
    really_small_R_roi.width = 665;
    really_small_R_roi.height = 371;

    // Sword example
    ROI sword_roi;
    sword_roi.name = "sword";
    sword_roi.start_x = 3328;
    sword_roi.start_y = 5949;
    sword_roi.width = 605;
    sword_roi.height = 844;

    // Flower example
    ROI flower_roi;
    flower_roi.name = "flower";
    flower_roi.start_x = 3404;
    flower_roi.start_y = 5455;
    flower_roi.width = 549;
    flower_roi.height = 469;

    // Double cross example
    ROI double_cross_roi;
    double_cross_roi.name = "double_cross";
    double_cross_roi.start_x = 3023;
    double_cross_roi.start_y = 5488;
    double_cross_roi.width = 325;
    double_cross_roi.height = 467;


    std::vector<ROI> regions_of_interest_test = {R_roi};

     for (const ROI& roi : regions_of_interest_test) {
         std::string example_dir = "/home/edvard/Documents/ReportExamplesTest/" + roi.name;
         Eigen::MatrixXd z = runWithRoi(K_pixel, roi);
         depth::saveDepthMap(z, example_dir, roi.name);
     }
    return 0;

    int roi_width = 900;
    int roi_height = 900;

    std::vector<ROI> regions_of_interest = getRois(image_width, image_height, roi_width, roi_height);
    for (const ROI& roi : regions_of_interest) {
        std::string patch_dir = "/home/edvard/dev/projects/cppPS/depthPatches";
        std::string patch_path = patch_dir + "/depth_" + std::to_string(roi.x) + "_" + std::to_string(roi.y) + ".dmap";
        if (fs::exists(patch_path)) {
            continue;
        }
        std::cout << "ROI at grid position (" << roi.x << ", " << roi.y
                  << ") starts at (" << roi.start_x << ", " << roi.start_y
                  << ") with size " << roi.width << "x" << roi.height << std::endl;


        Eigen::MatrixXd z = runWithRoi(K_pixel, roi);
        depth::saveDepthMap(z, "/home/edvard/dev/projects/cppPS/depthMapPatches",
                            std::to_string(roi.x) + "_" + std::to_string(roi.y));
        depth::saveDepthMapBinary(z.data(), roi, patch_dir);
    }

    // After generating and saving all the depth map patches, stitch them together
    std::string patch_dir = "/home/edvard/dev/projects/cppPS/depthPatches";
    Eigen::MatrixXd mega_depth_map = depth::stitchDepthMaps(patch_dir);
    depth::saveDepthMap(mega_depth_map, "/home/edvard/dev/projects/cppPS/megaDepthMap", "mega");

}

