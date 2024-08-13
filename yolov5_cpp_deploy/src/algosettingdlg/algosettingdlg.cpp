#include "algosettingdlg.h"
#include "ui_algosettingdlg.h"
#include "logger.h"
#include <QCheckBox>
#include <QDateTime>
#include <QDebug>
#include <QDir>
#include <QSettings>
#include <QMessageBox>
#include <QFileDialog>
#include <functional>
#include <future>
#include <QGraphicsScene>
#include <QMouseEvent>
#include <fstream>


using namespace std;
using namespace std::placeholders;

bool Node2Param(wikky_algo::CheckParam& checkparam, YAML::Node& _param)
{
	checkparam.th_conf = getValue<float>(_param, QString("Param_ModelConfig"), QString("th_conf"), 0.7);
	checkparam.th_nms = getValue<float>(_param, QString("Param_ModelConfig"), QString("th_nms"), 0.45);
	checkparam.th_obj = getValue<float>(_param, QString("Param_ModelConfig"), QString("th_obj"), 0.7);


	checkparam.ironcladScratchThreshold   = getValue<float>(_param, QString("Param_ModelConfig"), QString("ironcladScratchThreshold"), 30);
	checkparam.ironcladShinyMarkThreshold = getValue<float>(_param, QString("Param_ModelConfig"), QString("ironcladShinyMarkThreshold"), 900);
	checkparam.ironcladResiGlueThreshold  = getValue<float>(_param, QString("Param_ModelConfig"), QString("ironcladResiGlueThreshold"), 900);
	checkparam.ironcladDentThreshold      = getValue<float>(_param, QString("Param_ModelConfig"), QString("ironcladDentThreshold"), 900);
	checkparam.bootScratchThreshold       = getValue<float>(_param, QString("Param_ModelConfig"), QString("bootScratchThreshold"), 30);
	checkparam.bootDentThreshold          = getValue<float>(_param, QString("Param_ModelConfig"), QString("bootDentThreshold"), 900);
	checkparam.contaminationThreshold     = getValue<float>(_param, QString("Param_ModelConfig"), QString("contaminationThreshold"), 900);

	//======================================================================================================================================

	LOGD("Param_ModelConfig:");
	LOGD("Param_ModelConfig.th_conf   :{};", checkparam.th_conf);
	LOGD("Param_ModelConfig.th_nms    :{};", checkparam.th_nms);
	LOGD("Param_ModelConfig.th_obj    :{};", checkparam.th_obj);
	LOGD("Param_ModelConfig.ironcladScratchThreshold    :{};", checkparam.ironcladScratchThreshold);
	LOGD("Param_ModelConfig.ironcladShinyMarkThreshold  :{};", checkparam.ironcladShinyMarkThreshold);
	LOGD("Param_ModelConfig.ironcladResiGlueThreshold   :{};", checkparam.ironcladResiGlueThreshold);
	LOGD("Param_ModelConfig.ironcladDentThreshold       :{};", checkparam.ironcladDentThreshold);
	LOGD("Param_ModelConfig.bootScratchThreshold        :{};", checkparam.bootScratchThreshold);
	LOGD("Param_ModelConfig.bootDentThreshold           :{};", checkparam.bootDentThreshold);
	LOGD("Param_ModelConfig.contaminationThreshold      :{};", checkparam.contaminationThreshold);
	LOGD("\n");

	return true;
}

bool Param2Node(wikky_algo::CheckParam& checkparam, YAML::Node& _param)
{
	float _tmp;
	_tmp = getValue<float>(_param, QString("Param_ModelConfig"), QString("th_conf"), 0.7);
	if (_tmp != checkparam.th_conf) { LOGW("checkparam.th_conf: {} --> {};", _tmp, checkparam.th_conf); }

	_tmp = getValue<float>(_param, QString("Param_ModelConfig"), QString("th_nms"), 0.7);
	if (_tmp != checkparam.th_nms) { LOGW("checkparam.th_nms: {} --> {};", _tmp, checkparam.th_nms); }

	_tmp = getValue<float>(_param, QString("Param_ModelConfig"), QString("th_obj"), 0.7);
	if (_tmp != checkparam.th_obj) { LOGW("checkparam.th_obj: {} --> {};", _tmp, checkparam.th_obj); }

	_tmp = getValue<float>(_param, QString("Param_ModelConfig"), QString("ironcladScratchThreshold"), 0.7);
	if (_tmp != checkparam.ironcladScratchThreshold) { LOGW("checkparam.ironcladScratchThreshold: {} --> {};", _tmp, checkparam.ironcladScratchThreshold); }

	_tmp = getValue<float>(_param, QString("Param_ModelConfig"), QString("ironcladShinyMarkThreshold"), 0.7);
	if (_tmp != checkparam.ironcladShinyMarkThreshold) { LOGW("checkparam.ironcladShinyMarkThreshold: {} --> {};", _tmp, checkparam.ironcladShinyMarkThreshold); }

	_tmp = getValue<float>(_param, QString("Param_ModelConfig"), QString("ironcladResiGlueThreshold"), 0.7);
	if (_tmp != checkparam.ironcladResiGlueThreshold) { LOGW("checkparam.ironcladResiGlueThreshold: {} --> {};", _tmp, checkparam.ironcladResiGlueThreshold); }

	_tmp = getValue<float>(_param, QString("Param_ModelConfig"), QString("ironcladDentThreshold"), 0.7);
	if (_tmp != checkparam.ironcladDentThreshold) { LOGW("checkparam.ironcladDentThreshold: {} --> {};", _tmp, checkparam.ironcladDentThreshold); }

	_tmp = getValue<float>(_param, QString("Param_ModelConfig"), QString("bootScratchThreshold"), 0.7);
	if (_tmp != checkparam.bootScratchThreshold) { LOGW("checkparam.bootScratchThreshold: {} --> {};", _tmp, checkparam.bootScratchThreshold); }

	_tmp = getValue<float>(_param, QString("Param_ModelConfig"), QString("bootDentThreshold"), 0.7);
	if (_tmp != checkparam.bootDentThreshold) { LOGW("checkparam.bootDentThreshold: {} --> {};", _tmp, checkparam.bootDentThreshold); }

	_tmp = getValue<float>(_param, QString("Param_ModelConfig"), QString("contaminationThreshold"), 0.7);
	if (_tmp != checkparam.contaminationThreshold) { LOGW("checkparam.contaminationThreshold: {} --> {};", _tmp, checkparam.contaminationThreshold); }

	//======================================================================================================================================

	_param[QString("Param_ModelConfig").toStdString().c_str()][QString("th_conf").toStdString().c_str()]["value"] = checkparam.th_conf;
	_param[QString("Param_ModelConfig").toStdString().c_str()][QString("th_nms").toStdString().c_str()]["value"] = checkparam.th_nms;
	_param[QString("Param_ModelConfig").toStdString().c_str()][QString("th_obj").toStdString().c_str()]["value"] = checkparam.th_obj;

	_param[QString("Param_ModelConfig").toStdString().c_str()][QString("ironcladScratchThreshold").toStdString().c_str()]["value"]   = checkparam.ironcladScratchThreshold;
	_param[QString("Param_ModelConfig").toStdString().c_str()][QString("ironcladShinyMarkThreshold").toStdString().c_str()]["value"] = checkparam.ironcladShinyMarkThreshold;
	_param[QString("Param_ModelConfig").toStdString().c_str()][QString("ironcladResiGlueThreshold").toStdString().c_str()]["value"]  = checkparam.ironcladResiGlueThreshold;
	_param[QString("Param_ModelConfig").toStdString().c_str()][QString("ironcladDentThreshold").toStdString().c_str()]["value"]      = checkparam.ironcladDentThreshold;
	_param[QString("Param_ModelConfig").toStdString().c_str()][QString("bootScratchThreshold").toStdString().c_str()]["value"]       = checkparam.bootScratchThreshold;
	_param[QString("Param_ModelConfig").toStdString().c_str()][QString("bootDentThreshold").toStdString().c_str()]["value"]          = checkparam.bootDentThreshold;
	_param[QString("Param_ModelConfig").toStdString().c_str()][QString("contaminationThreshold").toStdString().c_str()]["value"]     = checkparam.contaminationThreshold;

	return true;
}

Qtalgosettingdlg::Qtalgosettingdlg(QWidget* parent)
	: QDialog(parent), ui(new Ui::algosettingdlg)
{
	ui->setupUi(this);
	this->installEventFilter(this);
	ui->gV_ShowImg->installEventFilter(this);
	connect(ui->treewidget, &QMyTreeWidget::TempSave, [=](QString objname, QString ves)
		{
			Node2Param(_tempparam, ui->treewidget->_mparam);
			//wikky_algo::SingleMat singlemat;
			//singlemat.imgori = _lastimg.clone();
			//_testcallback(singlemat, &_tempparam);
			//ui->gV_ShowImg->SetImage(singlemat.imgrst,false);
			m_bChanged = true;
			//ui.pB_Save->setEnabled(true);
		});

	Qt::WindowFlags windowFlag = Qt::Dialog;
	windowFlag |= Qt::WindowMinimizeButtonHint;
	windowFlag |= Qt::WindowMaximizeButtonHint;
	windowFlag |= Qt::WindowCloseButtonHint;
	setWindowFlags(windowFlag);

};

Qtalgosettingdlg::~Qtalgosettingdlg()
{
}

void Qtalgosettingdlg::SetLastParam(YAML::Node _param)
{
	ui->treewidget->LoadYAMLFile(_param);
}

void Qtalgosettingdlg::SetTestCallback(wikky_algo::TestCallback func)
{
	_testcallback = func;
}

void Qtalgosettingdlg::UpdatetoalgoImpl(wikky_algo::UpdateParam func)
{
	_testupdateparam = func;
}

void Qtalgosettingdlg::SetLastImage(cv::Mat img)
{
	_lastimg = img.clone();
	//_Qmap = QPixmap::fromImage(QImage((const uchar*)(_lastimg.data), _lastimg.cols, _lastimg.rows, _lastimg.cols * _lastimg.channels(), _lastimg.channels() == 3 ? QImage::Format_RGB888 : QImage::Format_Indexed8));
	//ui->gV_ShowImg->SetImage(_Qmap.toImage());
}

bool Qtalgosettingdlg::eventFilter(QObject* watched, QEvent* e)
{
	if (QEvent::Show == e->type()&& watched==this)
	{
		ui->gV_ShowImg->SetImage(_lastimg,true);
		return false;
	}
	if(QEvent::MouseButtonDblClick == e->type())
	{
		ui->gV_ShowImg->SetImage(_lastimg);
		return false;
	}
	if (QEvent::MouseButtonPress == e->type()&&Qt::RightButton == ((QMouseEvent*)e)->button())
	{
		if (_testcallback && !_lastimg.empty())
		{
			wikky_algo::SingleMat singlemat;
			singlemat.imgori = _lastimg.clone();
			Node2Param(_tempparam, ui->treewidget->_mparam);
			_testcallback(singlemat, &_tempparam);
			ui->gV_ShowImg->SetImage(singlemat.imgrst, false);
		}
		return false;
	}
	if (QEvent::Close == e->type())
	{
		if (m_bChanged && _testupdateparam)
		{
			if(QMessageBox::Yes == QMessageBox::warning(nullptr, "Param change warning", "Should save the param?", QMessageBox::Yes, QMessageBox::No))
				_testupdateparam(_tempparam);
		}
		return false;

	}
	return QDialog::eventFilter(watched, e);
}