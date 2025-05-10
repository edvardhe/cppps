#ifndef MATRIX_PROCESSING_H
#define MATRIX_PROCESSING_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include "optProblem.h"

struct ROI {
    int x;
    int y;
    int start_x;
    int start_y;
    int width;
    int height;
};

// Function declarations
Eigen::MatrixXd loadImagesToObservationMatrix(const std::string& directory_path,
                                              std::vector<std::string>& image_names,
                                              int start_x,
                                              int start_y,
                                              int roi_width,
                                              int roi_height);
std::vector<Eigen::Vector3d> loadJsonToPositionMatrix(const std::string& json_path,
                                                      const std::vector<std::string>& image_names);

Eigen::Matrix3d calculateCroppedKMatrix(Eigen::Matrix3d K,
                                        int start_x,
                                        int start_y);

void precomputeJacobian(PrecomputedData& data);

Eigen::Matrix3Xd KPixelToCm(Eigen::Matrix3Xd K_px,
                            float px_per_m_x,
                            float px_per_m_y);

Eigen::Matrix3Xd KCmToPixel(Eigen::Matrix3Xd K_cm,
                            float px_per_c_x,
                            float px_per_m_y);

void precomputeLightVectors(PrecomputedData& data,
                            double z_init);

#endif // MATRIX_PROCESSING_H