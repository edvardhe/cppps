#include "MatrixProcessing.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/core/eigen.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;



// Implementation of loadImagesToObservationMatrix
Eigen::MatrixXd loadImagesToObservationMatrix(const std::string& directory_path,
                                              std::vector<std::string>& image_names,
                                              int start_x,
                                              int start_y,
                                              int roi_width,
                                              int roi_height) {
    Eigen::MatrixXd eigen_images;
    std::vector<fs::path> valid_paths;
    int h = 0, w = 0, pixels = 0, index = 0;

    image_names.clear();

    for (const auto& entry : fs::directory_iterator(directory_path)) {
        auto ext = entry.path().extension().string();
        if (ext == ".JPG") {
            valid_paths.push_back(entry.path());
            image_names.push_back(entry.path().filename().string());
        }
    }

    cv::Mat color_image = cv::imread(valid_paths[0].string(), cv::IMREAD_COLOR);
    cv::Mat lab_image;
    cv::cvtColor(color_image, lab_image, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_channels;
    cv::split(lab_image, lab_channels);
    cv::Mat full_image = lab_channels[0];

    // Ensure ROI is within image bounds
    if (start_x + roi_width > full_image.cols || start_y + roi_height > full_image.rows) {
        throw std::runtime_error("ROI extends beyond image boundaries");
    }
    h = roi_height;
    w = roi_width;
    pixels = h * w;

    eigen_images = Eigen::MatrixXd(pixels, valid_paths.size());

    std::vector<Eigen::VectorXd> image_vectors(valid_paths.size());

    #pragma omp parallel for schedule(dynamic) num_threads(16)
    for (int i = 0; i < valid_paths.size(); ++i) {
        cv::Mat color_image = cv::imread(valid_paths[i].string(), cv::IMREAD_COLOR);
        cv::Mat lab_image;
        cv::cvtColor(color_image, lab_image, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> lab_channels;
        cv::split(lab_image, lab_channels);
        cv::Mat full_image = lab_channels[0];

        if (full_image.empty()) {
            std::cerr << "Warning: Could not read " << valid_paths[i] << std::endl;
            image_vectors[i] = Eigen::VectorXd::Zero(pixels);
            continue;
        }

        cv::Mat gray = full_image(cv::Rect(start_x, start_y, roi_width, roi_height));

        Eigen::MatrixXd eigen_matrix;
        cv::cv2eigen(gray, eigen_matrix);

        Eigen::VectorXd vec = Eigen::Map<Eigen::VectorXd>(eigen_matrix.data(), eigen_matrix.size());
        image_vectors[i] = vec / 255.0;
    }

    for (int i = 0; i < valid_paths.size(); ++i) {
        eigen_images.col(i) = image_vectors[i];
    }

    return eigen_images;
}

// Implementation of loadJsonToPositionMatrix
std::vector<Eigen::Vector3d> loadJsonToPositionMatrix(
        const std::string& json_path,
        const std::vector<std::string>& image_names,
        std::vector<Eigen::Vector3d> &light_dirs,
        Eigen::VectorXd &light_distances) {
    std::ifstream json_file(json_path);
    std::ifstream json_file2("/home/edvard/dev/projects/cppPS/light_positions1.json");
    if (!json_file.is_open()) {
        throw std::runtime_error("Could not open JSON file: " + json_path);
    }

    Eigen::Matrix3d R;
    R << 0.99902827, -0.0183341, -0.04007967,
     -0.01757252, -0.99965984,  0.01927205,
     -0.04041937, -0.01854902, -0.99901061;

    Eigen::Vector3d t;
    t << 33.10792131, 73.1994093, -217.6922226;

    json j;
    json_file >> j;

    json j2;
    json_file2 >> j2;

    Eigen::Matrix3d Rc;
    Rc << -1,  0,  0,
          0, 1,  0,
          0,  0, -1;

    size_t num_images = image_names.size();
    size_t position_dim = 3;
    std::vector<Eigen::Vector3d> positions = std::vector<Eigen::Vector3d>(num_images);

    light_dirs = std::vector<Eigen::Vector3d>(num_images);
    light_distances = Eigen::VectorXd(num_images);

    for (size_t i = 0; i < num_images; ++i) {
        const std::string& file_name = image_names[i];
        bool found = false;

        for (const auto& entry : j) {
            if (entry["image_name"] == file_name) {
                const auto& light_position_world = entry["light_position"];
                const auto& light_direction = entry["light_direction"];
                const auto& distance_to_sphere  = entry["distance_to_sphere"];
                Eigen::Vector3d pos_world(light_position_world[0],
                    light_position_world[1],
                    light_position_world[2]);

                Eigen::Vector3d dir(light_direction[0],
                    light_direction[1],
                    light_direction[2]);

                double distance = distance_to_sphere;

                Eigen::Vector3d sphere_pos(37.562,
                    31.016,
                    208.445);

                // Transform from world to camera coordinates
                // p_cam = R^T * (p_world - t)
                positions[i] = (sphere_pos + dir * distance) / 100.0; // Convert to meters
                //positions[i] = pos_world / 100.0; // Convert to meters

                light_dirs[i] = dir;
                light_distances[i] = distance/100.0;

                found = true;
                break;
            }
        }

        if (!found) {
            throw std::runtime_error("File name not found in JSON: " + file_name);
        }
    }

    return positions;
}

Eigen::Matrix3d calculateCroppedKMatrix(Eigen::Matrix3d K, int start_x, int start_y) {

    // Copy K to K_cropped
    Eigen::Matrix3d K_cropped_px = K;

    // Update the principal point in K_cropped
    K_cropped_px(0,2) -= start_x;
    K_cropped_px(1,2) -= start_y;

    return K_cropped_px;
}

Eigen::Matrix3Xd KPixelToCm(Eigen::Matrix3Xd K_px, float px_per_cm_x, float px_per_cm_y) {
    // Convert pixel coordinates to cm coordinates
    Eigen::Matrix3Xd K_cm = K_px;

    // Scale the intrinsic matrix by the pixel size
    K_cm(0, 0) /= px_per_cm_x;
    K_cm(1, 1) /= px_per_cm_y;

    // Adjust the principal point
    K_cm(0, 2) /= px_per_cm_x;
    K_cm(1, 2) /= px_per_cm_y;

    return K_cm;
}

void precomputeJacobian(PrecomputedData& data) {
    // Precompute the Jacobian for all pixels

    Eigen::Matrix3d K_px = data.K_pixel;

    int start_x = data.start_x;
    int start_y = data.start_y;

    int end_x = start_x + data.width;
    int end_y = start_y + data.height;

    double fx = K_px(0, 0);
    double fy = K_px(1, 1);
    double s = K_px(0, 1);
    double x0 = data.K(0, 2);
    double y0 = data.K(1, 2);

    data.J_all_pixels.resize(data.width * data.height);
    for (int x = start_x; x < end_x; ++x) {
        for (int y = start_y; y < end_y; ++y) {
            Eigen::Matrix3d J;
            int local_x = x - start_x;
            int local_y = y - start_y;
            J << fx,  s, -(local_x - x0),
                 0, fy, -(local_y - y0),
                 0,  0,  1;
            data.J_all_pixels[(y-start_y) + (x-start_x) * data.height] = J;
        }
    }
    return;
}


Eigen::Matrix3Xd KCmToPixel(Eigen::Matrix3Xd K_cm, float px_per_m_x, float px_per_m_y) {
        // Convert pixel coordinates to cm coordinates
        Eigen::Matrix3Xd K_px = K_cm;

        // Scale the intrinsic matrix by the pixel size
        K_px(0, 0) *= px_per_m_x;
        K_px(1, 1) *= px_per_m_y;
    
        // Adjust the principal point
        K_px(0, 2) *= px_per_m_x;
        K_px(1, 2) *= px_per_m_y;
    
        return K_px;
}