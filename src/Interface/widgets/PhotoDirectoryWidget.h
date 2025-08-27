//
// Created by edvard on 2025-06-09.
//

#ifndef PHOTODIRECTORYWIDGET_H
#define PHOTODIRECTORYWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QListWidget>
#include <QListWidgetItem>
#include <QString>
#include <QPixmap>
#include <QDir>
#include <QFileInfo>

class PhotoDirectoryWidget : public QWidget {
    Q_OBJECT

public:
    explicit PhotoDirectoryWidget(QWidget *parent = nullptr);
    void setPhotoDirectory(const QString &directoryPath);
    QString getPhotoDirectory() const;

private slots:
    void onDirectoryPathChanged();
    void refreshPhotoList();

private:
    void setupUI();
    void loadPhotosFromDirectory();
    QPixmap createThumbnail(const QString &imagePath, const QSize &size = QSize(100, 100));
    bool isImageFile(const QString &fileName);

    // UI Components
    QVBoxLayout *m_mainLayout;
    QHBoxLayout *m_pathLayout;
    QLabel *m_pathLabel;
    QLineEdit *m_pathLineEdit;
    QListWidget *m_photoListWidget;
    QPushButton *m_refreshButton;

    // Data
    QString m_currentDirectory;
    QStringList m_supportedFormats;
};

#endif //PHOTODIRECTORYWIDGET_H