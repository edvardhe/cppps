
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMenuBar>
#include "../ProblemConfig.h"

class QListWidget;
class QStackedWidget;
class PhotoDirectoryWidget;
class CameraParametersWidget;
class IndicatorSpheresWidget;
class ObjectParametersWidget;
class RegionsOfInterestWidget;
class ReviewAndSelectWidget;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    void setupMenuBar();
    void onNewProject();
    void onOpenProject();
    void onSidebarSelectionChanged(int index);
    void onSaveProject();
    void setupUi();
    void createContentWidgets();

    void syncConfigFromWidgets();
    void syncWidgetsFromConfig();

    void saveProjectConfig();
    void loadProjectConfig();
    QString getProjectConfigPath() const;

    ProblemConfig m_problemConfig;

    // Widgets
    PhotoDirectoryWidget* m_photoDirectoryWidget;
    CameraParametersWidget* m_cameraParametersWidget;
    IndicatorSpheresWidget* m_indicatorSpheresWidget;
    ObjectParametersWidget* m_objectParametersWidget;
    RegionsOfInterestWidget* m_regionsOfInterestWidget;
    ReviewAndSelectWidget* m_reviewAndSelectWidget;

    QListWidget* sidebarList;
    QStackedWidget* stackedWidget;
    static const int sidebarWidth = 180;

    // State variables
    bool m_projectLoaded;
    QString m_currentProjectPath;
    QString m_currentProjectName;
};

#endif // MAINWINDOW_H