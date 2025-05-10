#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <Eigen/Dense>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>
#include "MatrixProcessing.h"
#include <filesystem>
#include "depthMapHandler.h"

#include <opencv2/highgui.hpp>
//
// Created by edvard on 5/4/25.
//

namespace depth {

    void saveDepthMapAsImage(Eigen::MatrixXd depth_map, std::string output_dir, std::string file_name) {

        // Convert Eigen matrix to OpenCV Mat
        cv::Mat depth_map_cv(depth_map.rows(), depth_map.cols(), CV_64FC1);
        cv::eigen2cv(depth_map.eval(), depth_map_cv);

        // Find min and max values for normalization
        double min_val, max_val;
        cv::minMaxLoc(depth_map_cv, &min_val, &max_val);

        cv::Mat output;
        cv::normalize(depth_map_cv, output, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        // Save as both grayscale and color-mapped versions
        std::string grayscale_path = output_dir + "/" + file_name + "_depth.png";
        cv::imwrite(grayscale_path, output);

        // Create a color-mapped version for better visualization
        cv::Mat depth_color;
        cv::applyColorMap(output, depth_color, cv::COLORMAP_JET);

        std::string color_path = output_dir + "/" + file_name + "_depth_color.png";
        cv::imwrite(color_path, depth_color);

        std::cout << "Saved depth maps to " << grayscale_path << " and " << color_path << std::endl;
    }

    bool saveDepthMapAsObj(Eigen::MatrixXd depth_map, std::string output_dir, std::string file_name)
    {
        int width = depth_map.cols();
        int height = depth_map.rows();

        double scale_x = 1.0;
        double scale_y = static_cast<double>(height) / static_cast<double>(width);
        double scale_z = 1.0;

        std::string filename = output_dir + file_name;

        std::ofstream objFile(filename);
        if (!objFile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return false;
        }

        // Write header
        objFile << "# Depth map exported as OBJ\n";
        objFile << "# Width: " << width << ", Height: " << height << "\n";

        // Map depth data
        cv::Mat cv_depth(height, width, CV_64FC1);
        cv::eigen2cv(depth_map, cv_depth);

        // Find min/max depth for normalization
        double min_depth, max_depth;
        cv::minMaxLoc(cv_depth, &min_depth, &max_depth);
        std::cout << "Depth range: " << min_depth << " to " << max_depth << std::endl;

        // Write vertices
        // The coordinate system is: X right, Y up, Z backward (toward viewer)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                double depth = cv_depth.at<double>(x, y);

                // Normalize x and y to [-1, 1] range
                double norm_x = (2.0 * x / (width - 1) - 1.0) * scale_x;
                double norm_y = (1.0 - 2.0 * y / (height - 1)) * scale_y; // Flip Y to match 3D convention
                double norm_z = 0;
                if ((max_depth - min_depth)>0) {
                    norm_z = ((depth - min_depth) / (max_depth - min_depth)) * scale_z;
                }

                objFile << "v " << norm_x << " " << norm_y << " "
                       << norm_z << "\n";
            }
        }

        // Write faces (triangles)
        for (int y = 0; y < height - 1; y++) {
            for (int x = 0; x < width - 1; x++) {
                // Get the four corners of this grid cell (1-indexed for OBJ format)
                int topLeft = (y * width + x) + 1;
                int bottomLeft = ((y+1) * width + x) + 1;
                int topRight = (y * width + (x+1)) + 1;
                int bottomRight = ((y+1) * width + (x+1)) + 1;

                // Write two triangles with counter-clockwise winding
                // First triangle: top-left, bottom-left, bottom-right
                objFile << "f " << topLeft << " " << bottomLeft << " " << bottomRight << "\n";

                // Second triangle: top-left, bottom-right, top-right
                objFile << "f " << topLeft << " " << bottomRight << " " << topRight << "\n";
            }
        }



        objFile.close();
        //std::cout << "Depth map saved as OBJ file: " << filename << std::endl;
        return true;
    }

    void saveDepthMap(Eigen::MatrixXd depth_map, std::string output_dir, std::string file_name) {

        std::string image_directory = output_dir + "/depthMap/";
        std::string image_filename = file_name + ".png";

        std::string obj_directory = output_dir + "/obj/";
        std::string obj_filename = file_name + ".obj";

        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }

        if (!std::filesystem::exists(image_directory)) {
            std::filesystem::create_directories(output_dir + "/depthMap");
        }

        if (!std::filesystem::exists(obj_directory)) {
            std::filesystem::create_directories(output_dir + "/obj");
        }

        saveDepthMapAsImage(depth_map, image_directory, image_filename);
        saveDepthMapAsObj(depth_map, obj_directory, obj_filename);
    }

    // Save depth map as a custom binary format
    bool saveDepthMapBinary(const double* depth_data, const ROI& roi, const std::string& directory, int overlap) {
        // Create filename based on ROI grid position
        std::string filename = directory + "/depth_" +
                              std::to_string(roi.x) + "_" +
                              std::to_string(roi.y) + ".dmap";

        // Calculate min/max depth values (for metadata)
        double min_depth = std::numeric_limits<double>::max();
        double max_depth = std::numeric_limits<double>::lowest();

        for (int i = 0; i < roi.width * roi.height; ++i) {
            // Skip invalid depths (use a small positive threshold)
            if (depth_data[i] > 0.00001) {
                min_depth = std::min(min_depth, depth_data[i]);
                max_depth = std::max(max_depth, depth_data[i]);
            }
        }

        // If no valid depths were found
        if (min_depth == std::numeric_limits<double>::max()) {
            min_depth = 0.0;
            max_depth = 0.0;
        }

        // Create and initialize header
        DepthMapHeader header;
        std::memcpy(header.magic, "DMAP", 4);
        header.version = 1;
        header.roi_x = roi.x;
        header.roi_y = roi.y;
        header.start_x = roi.start_x;
        header.start_y = roi.start_y;
        header.width = roi.width;
        header.height = roi.height;
        header.overlap = overlap;
        header.min_depth = min_depth;
        header.max_depth = max_depth;
        std::memset(header.reserved, 0, sizeof(header.reserved));

        try {
            // Open binary file for writing
            std::ofstream file(filename, std::ios::binary);
            if (!file) {
                std::cerr << "Failed to open file for writing: " << filename << std::endl;
                return false;
            }

            // Write header
            file.write(reinterpret_cast<const char*>(&header), sizeof(DepthMapHeader));

            // Write depth data
            file.write(reinterpret_cast<const char*>(depth_data),
                      roi.width * roi.height * sizeof(double));

            file.close();

            std::cout << "Successfully wrote depth map to: " << filename << std::endl;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error saving depth map: " << e.what() << std::endl;
            return false;
        }
    }

    bool loadDepthMapBinary(const std::string& filename, double*& depth_data, ROI& roi) {
        try {
            // Open binary file for reading
            std::ifstream file(filename, std::ios::binary);
            if (!file) {
                std::cerr << "Failed to open file for reading: " << filename << std::endl;
                return false;
            }

            // Read header
            DepthMapHeader header;
            file.read(reinterpret_cast<char*>(&header), sizeof(DepthMapHeader));

            // Verify magic identifier
            if (std::memcmp(header.magic, "DMAP", 4) != 0) {
                std::cerr << "Invalid file format: " << filename << std::endl;
                return false;
            }

            // Set ROI data
            roi.x = header.roi_x;
            roi.y = header.roi_y;
            roi.start_x = header.start_x;
            roi.start_y = header.start_y;
            roi.width = header.width;
            roi.height = header.height;

            // Allocate memory for depth data
            depth_data = new double[roi.width * roi.height];

            // Read depth data
            file.read(reinterpret_cast<char*>(depth_data),
                     roi.width * roi.height * sizeof(double));

            file.close();

            std::cout << "Successfully read depth map from: " << filename << std::endl;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading depth map: " << e.what() << std::endl;
            return false;
        }
    }

    Eigen::MatrixXd stitchDepthMaps(const std::string& directory) {
        // List all .dmap files in the directory
        std::vector<std::string> dmap_files;

        std::string debug_dir = "/home/edvard/dev/projects/cppPS/debugImages/stitching";
        std::filesystem::create_directories(debug_dir);


        try {
            for (const auto& entry : std::filesystem::directory_iterator(directory)) {
                if (entry.path().extension() == ".dmap") {
                    dmap_files.push_back(entry.path().string());
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading directory: " << e.what() << std::endl;
            return Eigen::MatrixXd();
        }

        if (dmap_files.empty()) {
            std::cerr << "No .dmap files found in directory: " << directory << std::endl;
            return Eigen::MatrixXd();
        }

        // First, determine the full dimensions by finding max coordinates
        int max_x = 0, max_y = 0;
        int total_width = 0, total_height = 0;


        for (const auto& filename : dmap_files) {
            double* temp_data = nullptr;
            ROI roi;

            if (loadDepthMapBinary(filename, temp_data, roi)) {
                // Update maximum coordinates
                int right_edge = roi.start_x + roi.width;
                int bottom_edge = roi.start_y + roi.height;

                total_width = std::max(total_width, right_edge);
                total_height = std::max(total_height, bottom_edge);

                max_x = std::max(max_x, roi.x);
                max_y = std::max(max_y, roi.y);

                // Free temporary data
                delete[] temp_data;
            }
        }

        // Create the mega matrix with the determined dimensions
        Eigen::MatrixXd mega_depth_map = Eigen::MatrixXd::Zero(total_height, total_width);

        int patch_count = 0;

        // Load and place each depth map into the mega matrix
        for (const auto& filename : dmap_files) {
            double* depth_data = nullptr;
            ROI roi;

            if (loadDepthMapBinary(filename, depth_data, roi)) {
                // Place the depth data in the correct position in the mega matrix

                Eigen::Map<Eigen::MatrixXd> depth_matrix(depth_data, roi.width, roi.height);
                cv::Mat depth_map_cv;
                cv::eigen2cv(depth_matrix.eval(), depth_map_cv);

                cv::Mat depth_viz(roi.height, roi.width, CV_64FC1);
                cv::normalize(depth_map_cv, depth_viz, 0, 255, cv::NORM_MINMAX, CV_8UC1);

                for (int y = 0; y < roi.height; ++y) {
                    for (int x = 0; x < roi.width; ++x) {
                        int global_x = roi.start_x + x;
                        int global_y = roi.start_y + y;

                        // Calculate overlap weight for blending (simple approach: average in overlap zones)
                        double depth_value = depth_map_cv.at<double>(y,x);

                        // Check if this is a valid depth
                        if (depth_value > 0.00001) {
                            // If there's an existing value in the mega matrix, blend them
                            if (mega_depth_map(global_y, global_x) > 0.00001) {
                                // Simple blending: average the values
                                mega_depth_map.coeffRef(global_y, global_x) =
                                    (mega_depth_map.coeffRef(global_y, global_x) + depth_value) / 2.0;
                            } else {
                                // No existing value, just set it
                                mega_depth_map.coeffRef(global_y, global_x) = depth_value;
                            }
                        }
                    }
                }

                // // Convert the Eigen array to an OpenCV Mat for visualization
                // cv::Mat depth_viz(total_height, total_width, CV_64FC1);
                // cv::eigen2cv(mega_depth_map, depth_viz);
                //
                // // First, clamp the depth values to a desired range
                // double min_clamp = 2.146;
                //
                // // Create a copy for clamping (to not modify the original)
                // cv::Mat clamped_depth;
                // cv::max(depth_viz, min_clamp, clamped_depth);  // Any value below min_clamp will be set to min_clamp
                //
                // // Find min and max for proper normalization
                // double min_val, max_val;
                //
                // // Normalize and convert to 8-bit for visualization
                // cv::Mat depth_norm, depth_color;
                //
                // // Convert to 8-bit with normalization
                // cv::normalize(clamped_depth, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                //
                // cv::minMaxLoc(depth_norm, &min_val, &max_val);
                //
                // // Apply color map for better visualization
                // cv::applyColorMap(depth_norm, depth_color, cv::COLORMAP_JET);
                //
                // // Add text showing patch count and position
                // std::string text = "Patch: " + std::to_string(patch_count) +
                //                   " Pos: (" + std::to_string(roi.x) + "," +
                //                   std::to_string(roi.y) + ")";
                // cv::putText(depth_color, text, cv::Point(20, 30),
                //            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
                //
                // // Save the debug image
                // std::string debug_filename = debug_dir + "/depth_map_patch_" +
                //                             std::to_string(patch_count) + ".png";
                // cv::imwrite(debug_filename, depth_color);

                patch_count++;

                // Free the depth data
                delete[] depth_data;
            }
        }

        std::cout << "Successfully stitched " << dmap_files.size() << " depth maps into a "
                  << total_height << " x " << total_width << " matrix" << std::endl;

        return mega_depth_map;
    }


}
#include "depthMapHandler.h"
