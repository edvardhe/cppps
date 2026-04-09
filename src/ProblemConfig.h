#pragma once

#include <QString>
#include <QVector>
#include <QJsonObject>
#include <QJsonArray>
#include <Eigen/Dense>
#include <vector>

struct ROIConfig {
    QString name;
    int startX = 0;
    int startY = 0;
    int width = 0;
    int height = 0;
    int gridX = 0;
    int gridY = 0;

    QJsonObject toJson() const {
        QJsonObject obj;
        obj["name"] = name;
        obj["startX"] = startX;
        obj["startY"] = startY;
        obj["width"] = width;
        obj["height"] = height;
        obj["gridX"] = gridX;
        obj["gridY"] = gridY;
        return obj;
    }

    static ROIConfig fromJson(const QJsonObject& obj) {
        ROIConfig config;
        config.name = obj["name"].toString();
        config.startX = obj["startX"].toInt();
        config.startY = obj["startY"].toInt();
        config.width = obj["width"].toInt();
        config.height = obj["height"].toInt();
        config.gridX = obj["gridX"].toInt();
        config.gridY = obj["gridY"].toInt();
        return config;
    }
};

struct CameraConfig {
    double focalLengthMm = 0.0;
    double sensorWidthMm = 0.0;
    double sensorHeightMm = 0.0;
    int imageWidthPx = 0;
    int imageHeightPx = 0;
    double principalPointOffsetX = 0.0;
    double principalPointOffsetY = 0.0;

    QJsonObject toJson() const {
        QJsonObject obj;
        obj["focalLengthMm"] = focalLengthMm;
        obj["sensorWidthMm"] = sensorWidthMm;
        obj["sensorHeightMm"] = sensorHeightMm;
        obj["imageWidthPx"] = imageWidthPx;
        obj["imageHeightPx"] = imageHeightPx;
        obj["principalPointOffsetX"] = principalPointOffsetX;
        obj["principalPointOffsetY"] = principalPointOffsetY;
        return obj;
    }

    static CameraConfig fromJson(const QJsonObject& obj) {
        CameraConfig config;
        config.focalLengthMm = obj["focalLengthMm"].toDouble();
        config.sensorWidthMm = obj["sensorWidthMm"].toDouble();
        config.sensorHeightMm = obj["sensorHeightMm"].toDouble();
        config.imageWidthPx = obj["imageWidthPx"].toInt();
        config.imageHeightPx = obj["imageHeightPx"].toInt();
        config.principalPointOffsetX = obj["principalPointOffsetX"].toDouble();
        config.principalPointOffsetY = obj["principalPointOffsetY"].toDouble();
        return config;
    }
};

struct SphereMarker {
    QString name;
    double x = 0.0;  // Pixel coordinates in reference image
    double y = 0.0;
    double radius = 20.0;  // Visual radius for the marker

    QJsonObject toJson() const {
        QJsonObject obj;
        obj["name"] = name;
        obj["x"] = x;
        obj["y"] = y;
        obj["radius"] = radius;
        return obj;
    }

    static SphereMarker fromJson(const QJsonObject& obj) {
        SphereMarker marker;
        marker.name = obj["name"].toString();
        marker.x = obj["x"].toDouble();
        marker.y = obj["y"].toDouble();
        marker.radius = obj["radius"].toDouble(20.0);
        return marker;
    }
};

struct IndicatorSphereConfig {
    QString referenceImagePath;
    QVector<SphereMarker> sphereMarkers;

    // Legacy 3D data (kept for backward compatibility)
    Eigen::Vector3d spherePosition = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> lightPositions;
    std::vector<Eigen::Vector3d> lightDirections;
    double lightIntensity = 1.0;

    QJsonObject toJson() const {
        QJsonObject obj;
        obj["referenceImagePath"] = referenceImagePath;

        QJsonArray markersArray;
        for (const auto& marker : sphereMarkers) {
            markersArray.append(marker.toJson());
        }
        obj["sphereMarkers"] = markersArray;

        // Legacy 3D data
        QJsonArray spherePos;
        spherePos.append(spherePosition.x());
        spherePos.append(spherePosition.y());
        spherePos.append(spherePosition.z());
        obj["spherePosition"] = spherePos;

        QJsonArray lightPos;
        for (const auto& pos : lightPositions) {
            QJsonArray p;
            p.append(pos.x());
            p.append(pos.y());
            p.append(pos.z());
            lightPos.append(p);
        }
        obj["lightPositions"] = lightPos;

        QJsonArray lightDir;
        for (const auto& dir : lightDirections) {
            QJsonArray d;
            d.append(dir.x());
            d.append(dir.y());
            d.append(dir.z());
            lightDir.append(d);
        }
        obj["lightDirections"] = lightDir;
        obj["lightIntensity"] = lightIntensity;
        return obj;
    }

    static IndicatorSphereConfig fromJson(const QJsonObject& obj) {
        IndicatorSphereConfig config;

        config.referenceImagePath = obj["referenceImagePath"].toString();

        QJsonArray markersArray = obj["sphereMarkers"].toArray();
        for (const auto& m : markersArray) {
            config.sphereMarkers.append(SphereMarker::fromJson(m.toObject()));
        }

        // Legacy 3D data
        QJsonArray spherePos = obj["spherePosition"].toArray();
        if (spherePos.size() == 3) {
            config.spherePosition = Eigen::Vector3d(
                spherePos[0].toDouble(),
                spherePos[1].toDouble(),
                spherePos[2].toDouble()
            );
        }

        QJsonArray lightPos = obj["lightPositions"].toArray();
        for (const auto& p : lightPos) {
            QJsonArray pos = p.toArray();
            if (pos.size() == 3) {
                config.lightPositions.push_back(Eigen::Vector3d(
                    pos[0].toDouble(),
                    pos[1].toDouble(),
                    pos[2].toDouble()
                ));
            }
        }

        QJsonArray lightDir = obj["lightDirections"].toArray();
        for (const auto& d : lightDir) {
            QJsonArray dir = d.toArray();
            if (dir.size() == 3) {
                config.lightDirections.push_back(Eigen::Vector3d(
                    dir[0].toDouble(),
                    dir[1].toDouble(),
                    dir[2].toDouble()
                ));
            }
        }

        config.lightIntensity = obj["lightIntensity"].toDouble(1.0);
        return config;
    }
};

struct ObjectConfig {
    double initialDepth = 0.0;
    double initialAlbedo = 1.0;

    QJsonObject toJson() const {
        QJsonObject obj;
        obj["initialDepth"] = initialDepth;
        obj["initialAlbedo"] = initialAlbedo;
        return obj;
    }

    static ObjectConfig fromJson(const QJsonObject& obj) {
        ObjectConfig config;
        config.initialDepth = obj["initialDepth"].toDouble();
        config.initialAlbedo = obj["initialAlbedo"].toDouble(1.0);
        return config;
    }
};

struct ProblemConfig {
    QString photoDirectory;
    CameraConfig camera;
    IndicatorSphereConfig indicators;
    ObjectConfig object;
    QVector<ROIConfig> rois;
    int selectedRoiIndex = -1;

    QJsonObject toJson() const {
        QJsonObject obj;
        obj["photoDirectory"] = photoDirectory;
        obj["camera"] = camera.toJson();
        obj["indicators"] = indicators.toJson();
        obj["object"] = object.toJson();

        QJsonArray roisArray;
        for (const auto& roi : rois) {
            roisArray.append(roi.toJson());
        }
        obj["rois"] = roisArray;
        obj["selectedRoiIndex"] = selectedRoiIndex;
        return obj;
    }

    static ProblemConfig fromJson(const QJsonObject& obj) {
        ProblemConfig config;
        config.photoDirectory = obj["photoDirectory"].toString();
        config.camera = CameraConfig::fromJson(obj["camera"].toObject());
        config.indicators = IndicatorSphereConfig::fromJson(obj["indicators"].toObject());
        config.object = ObjectConfig::fromJson(obj["object"].toObject());

        QJsonArray roisArray = obj["rois"].toArray();
        for (const auto& roi : roisArray) {
            config.rois.append(ROIConfig::fromJson(roi.toObject()));
        }
        config.selectedRoiIndex = obj["selectedRoiIndex"].toInt(-1);
        return config;
    }
};
