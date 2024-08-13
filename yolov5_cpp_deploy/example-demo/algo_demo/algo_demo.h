#pragma once

#include <intsafe.h>

#include <QDir>
#include <QHeaderView>
#include <QSettings>
#include "ui_algo_demo.h"
#include <intsafe.h>
#include "opencv2/opencv.hpp"
#include <QMetaType>
#include <QLibrary>
#include "windows.h"
#include "ibasealgorithm.h"
#include "imagewatch.h"

class Qtalgo_demo : public QMainWindow
{
	Q_OBJECT

private:
	wikky_algo::pExportALG CreateLocalCamera;
	wikky_algo::pDeleteALG DeleteCamera;
	QString AppPath;
	QString m_sImageListPath;
	int m_iFormat = -1;
	QSettings* configIniRead = nullptr;
	LARGE_INTEGER nFreq;
	cv::Mat ImgRead, m_matCheck, m_tempshow, m_tempRGB;
	QImage disImage;
	QImage disImage_ori;
	QTimer* m_Tautorun = nullptr;
	int m_w, m_h, m_c;
	float fInterval;
	int m_ichecktimes;
	bool m_bContinueCheck = false;
	bool m_bNGSTOP = false;
	bool m_bOKSTOP = false;
	int m_iReadInded = 0;
public:
	Qtalgo_demo(QWidget* parent = Q_NULLPTR);
	void connectsignal();
	cv::Mat ReadImage(QString);
	wikky_algo::IBaseAlg* _CheckClass = nullptr;
	void ShowMatToLabel(const cv::Mat img, const cv::Mat img_ori);
	void resizeEvent(QResizeEvent*);
	void showEvent(QShowEvent*);
	~Qtalgo_demo();
private:
	Ui::algo_demo ui;
	void initImageLS(QString str);
	bool containImages(QDir&);
	bool isImage(QFileInfo& info);
	ImageWidget* m_Image = nullptr;
	QString currentmodule;
	bool copyDirectoryFiles(const QString& fromDir, const QString& toDir, bool coverFileIfExist);
public slots:
	void onSelectImageList(QListWidgetItem* item, QListWidgetItem* it);
	void onInitAlgo();
	void onUpdateAlgo();
	void recvShowPicSignal(QImage image, const cv::Mat& img_ori);//接收并显示图片的函数
};
