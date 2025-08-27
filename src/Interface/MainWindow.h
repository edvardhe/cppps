
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMenuBar>

class QListWidget;
class QStackedWidget;
class PhotoDirectoryWidget;

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

    // Widgets
    PhotoDirectoryWidget* m_photoDirectoryWidget;

    QListWidget* sidebarList;
    QStackedWidget* stackedWidget;
    static const int sidebarWidth = 180;

    // State variables
    bool m_projectLoaded;
    QString m_currentProjectPath;
    QString m_currentProjectName;
};

#endif // MAINWINDOW_H