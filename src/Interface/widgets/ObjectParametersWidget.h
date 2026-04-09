//
// Created by edvard on 2025-06-09.
//

#ifndef OBJECTPARAMETERSWIDGET_H
#define OBJECTPARAMETERSWIDGET_H



#include <QWidget>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include "../../ProblemConfig.h"

class ObjectParametersWidget : public QWidget {
    Q_OBJECT

public:
    explicit ObjectParametersWidget(QWidget *parent = nullptr);

    ObjectConfig config() const;
    void setConfig(const ObjectConfig& config);

signals:
    void objectConfigChanged(const ObjectConfig& config);

private slots:
    void onAnyValueChanged();

private:
    void setupUI();

    QDoubleSpinBox* m_initialDepth;
    QDoubleSpinBox* m_initialAlbedo;
};


#endif //OBJECTPARAMETERSWIDGET_H
