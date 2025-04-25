#include "MatrixProcessing.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <vector>

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
    int h = 0, w = 0, pixels = 0, index = 0;

    image_names.clear();

    for (const auto& entry : fs::directory_iterator(directory_path)) {
        auto ext = entry.path().extension().string();
        if (ext == ".PNG" || ext == ".JPG") {
            image_names.push_back(entry.path().filename().string());
            cv::Mat full_image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

            if (h == 0 && w == 0) {
                // Ensure ROI is within image bounds
                if (start_x + roi_width > full_image.cols || start_y + roi_height > full_image.rows) {
                    throw std::runtime_error("ROI extends beyond image boundaries");
                }
                h = roi_height;
                w = roi_width;
                pixels = h * w;
                eigen_images = Eigen::MatrixXd(pixels, std::distance(fs::directory_iterator(directory_path), fs::directory_iterator()));
            }

            if (full_image.empty()) {
                std::cerr << "Warning: Could not read " << entry.path() << std::endl;
                continue;
            }

            // Extract ROI
            cv::Mat gray = full_image(cv::Rect(start_x, start_y, roi_width, roi_height));
            
            gray.convertTo(gray, CV_64F, 1.0 / 255.0);
            Eigen::VectorXd vec(gray.rows * gray.cols);
            for (int r = 0; r < gray.rows; ++r) {
                for (int c = 0; c < gray.cols; ++c) {
                    vec(r * gray.cols + c) = gray.at<double>(r, c);
                }
            }

            eigen_images.col(index++) = vec;
        }
    }

    if (eigen_images.rows() == 0 || eigen_images.cols() == 0) {
        throw std::runtime_error("No valid images found in " + directory_path);
    }

    return eigen_images;
}

// Implementation of loadJsonToPositionMatrix
std::vector<Eigen::Vector3d> loadJsonToPositionMatrix(const std::string& json_path, const std::vector<std::string>& image_names) {
    std::ifstream json_file(json_path);
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

    Eigen::Matrix3d Rc;
    Rc << -1,  0,  0,
          0, 1,  0,
          0,  0, -1;

    size_t num_images = image_names.size();
    size_t position_dim = 3;
    std::vector<Eigen::Vector3d> positions = std::vector<Eigen::Vector3d>(num_images);

    for (size_t i = 0; i < num_images; ++i) {
        const std::string& file_name = image_names[i];
        bool found = false;

        for (const auto& entry : j) {
            if (entry["image_name"] == file_name) {
                const auto& light_position_world = entry["light_position"];
                
                Eigen::Vector3d pos_world(light_position_world[0],
                    light_position_world[1],
                    light_position_world[2]);

                // Transform from world to camera coordinates
                // p_cam = R^T * (p_world - t)
                
                
                positions[i] = (Rc * (R.transpose() * (pos_world - t))) / 100.0; // Convert to meters

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

void precomputeJacobian(PrecomputedData& data) {
    // Precompute the Jacobian for all pixels

    Eigen::Matrix3d K_px = data.K;

    int start_x = data.start_x;
    int start_y = data.start_y;

    int end_x = start_x + data.width;
    int end_y = start_y + data.height;

    double fx = K_px(0, 0);
    double fy = K_px(1, 1);
    double s = K_px(0, 1);
    double x0 = K_px(0, 2);
    double y0 = K_px(1, 2);

    data.J_all_pixels.resize(data.width * data.height);

    for (int y = start_y; y < end_y; ++y) {
        for (int x = start_x; x < end_x; ++x) {
            Eigen::Matrix3d J;
            J << fx,  s, -(x - x0),
                0, fy, -(y - y0),
                0,  0,  1;
            data.J_all_pixels[(y-start_y) * data.width + (x-start_x)] = J;
        }
    }

    return;
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

void precomputeLightVectors(PrecomputedData& data, double z_init) {

    int num_pixels = data.I.rows();
    int num_lights = data.I.cols();

    data.light_dirs.resize(num_pixels * num_lights);
    data.light_distances.resize(num_pixels * num_lights);
    data.anisotropy.resize(num_pixels * num_lights);

    for (int j = 0; j < num_pixels; ++j) {
        int x_j = j % data.width;
        int y_j = j / data.width;
        // Back-project pixel j using initial depth (z_init[j])
        Eigen::Vector3d ray_dir = data.K.inverse() * Eigen::Vector3d(x_j, y_j, 1.0);
        Eigen::Vector3d x_j_3D = z_init * ray_dir;

        for (int i = 0; i < num_lights; ++i) {
            int idx = j * num_lights + i;
            Eigen::Vector3d light_dir = x_j_3D - data.light_positions[i];
            data.light_distances[idx] = light_dir.norm();
            data.light_dirs[idx] = light_dir.normalized();

            double anisotropy = pow(light_dir.dot(data.n_s_i[i]), data.mu_i[i]);

            data.anisotropy[idx] = anisotropy;

        }
    }
}