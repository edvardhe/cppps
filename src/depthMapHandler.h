//
// Created by edvard on 5/4/25.
//

#ifndef DEPTHMAPHANDLER_H
#define DEPTHMAPHANDLER_H

#include <string>
#include "MatrixProcessing.h"

namespace depth {

    // Header structure for our binary format
    struct DepthMapHeader {
        char magic[4];      // "DMAP" magic identifier
        uint32_t version;   // Format version (1)
        uint32_t roi_x;     // Grid position X
        uint32_t roi_y;     // Grid position Y
        uint32_t start_x;   // Pixel start X
        uint32_t start_y;   // Pixel start Y
        uint32_t width;     // Width in pixels
        uint32_t height;    // Height in pixels
        uint32_t overlap;   // Overlap size
        double min_depth;   // Minimum depth value
        double max_depth;   // Maximum depth value
        char reserved[64];  // Reserved for future use
    };

    void saveDepthMapAsImage(Eigen::MatrixXd depth_map, std::string output_dir, std::string file_name);

    bool saveDepthMapAsObj(Eigen::MatrixXd depth_map, std::string output_dir, std::string file_name);

    void saveDepthMap(Eigen::MatrixXd depth_map, std::string output_dir, std::string file_name);

    bool saveDepthMapBinary(const double* depth_data, const ROI& roi, const std::string& directory, int overlap = 10);

    bool loadDepthMapBinary(const std::string& filename, double*& depth_data, ROI& roi);

    Eigen::MatrixXd stitchDepthMaps(const std::string& directory);
}

#endif //DEPTHMAPHANDLER_H