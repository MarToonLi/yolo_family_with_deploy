#pragma once
#include <QtCore>
#include <QTreeWidget>
#include <QObject>
#include "yaml-cpp/yaml.h"
#include "yaml-cpp/parser.h"
#include "yaml-cpp/eventhandler.h"
#include "yaml-cpp/yaml.h"

#include <QTextEdit>
#include <QTimer>
#include <QSlider>
#include <QLineEdit>


class QMyTextEdit : public QTextEdit
{
	Q_OBJECT
public:
	QMyTextEdit(QWidget* parent)
		: QTextEdit(parent)
	{
	}
private:
	QTimer* times = nullptr;
public slots:
	void showEvent(QShowEvent*)
	{
		times = new QTimer(this);
		times->setSingleShot(true);
		connect(times, &QTimer::timeout, [=]()
			{
				this->setVisible(false);
			});
		times->start(2000);
	}
};

class QMySlider : public QSlider
{
	Q_OBJECT
public:
	QMySlider(Qt::Orientation orientation, QWidget* parent = nullptr)
		: QSlider(orientation, parent)
	{
	}
	void wheelEvent(QWheelEvent* e)
	{
		QWidget::wheelEvent(e);
		//备注，千万别使用QSlider::wheelEvent(e);否则等于没重写
	}
};

class QMyTreeWidget : public QTreeWidget
{
	Q_OBJECT
signals:
	void TempSave(QString, QString);
	void Signal_DrawRegion(QString);
private:
	QTextDocument* document = nullptr;
	QTextEdit* editor = nullptr;
	QMyTextEdit* detailtext = nullptr;
	QString m_ErrorNode;
public:
	YAML::Node _mparam;
	QMyTreeWidget(QWidget* parent);
	~QMyTreeWidget();
	bool LoadYAMLFile(YAML::Node);
	bool SaveYAMLFile(QString filepath);
	void expandbyString();
protected:
	bool eventFilter(QObject*, QEvent*);
public slots:
	void SliderValueChanged(int);
	void ComboValueChanged(int);
	void CheckValueChanged(int);
	void CheckChanged(int);
	void ClickPushButton();
	void LineValueChanged(QString);
	void showCnDetail(QTreeWidgetItem*, int);
	void textAreaChanged();
};
