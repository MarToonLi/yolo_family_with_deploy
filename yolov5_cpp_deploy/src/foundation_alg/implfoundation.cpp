#include "foundation.h"
#include <iostream>
#include <QSettings>
#include <QCoreapplication>
#include "implfoundation.h"
#include "logger.h"
#include <functional>
#include <future>
#include <fstream>
#include <filesystem>


namespace wikky_algo
{
	std::string d2str(double val, int precision) {
		std::ostringstream stream;
		stream << std::fixed << std::setprecision(precision) << val;
		std::string result = stream.str();
		return result;
	}

	std::string vectors2string(std::vector<std::string> vectors) {
		std::ostringstream oss;
		for (size_t i = 0; i < vectors.size(); ++i) {
			oss << vectors[i];
			if (i != vectors.size() - 1) { oss << "-"; }  // 如果不是最后一个数字，则添加横杠作为分隔符
		}
		std::string vstring = oss.str();

		return vstring;
	}

	std::string vectors2string(std::vector<int> vectors) {
		std::ostringstream oss;
		for (size_t i = 0; i < vectors.size(); ++i) {
			oss << std::to_string(vectors[i]);
			if (i != vectors.size() - 1) { oss << "-"; }  // 如果不是最后一个数字，则添加横杠作为分隔符
		}
		std::string vstring = oss.str();

		return vstring;
	}

	std::string vectors2string(std::vector<BoxInfo> vectors) {
		if (vectors.size() == 0) { return ""; }

		std::ostringstream oss;
		for (size_t i = 0; i < vectors.size(); ++i) {
			oss << vectors[i].label;
			oss << "(";
			oss << vectors[i].score;
			oss << ")";
			if (i != vectors.size() - 1) { oss << "-"; }  // 如果不是最后一个数字，则添加横杠作为分隔符
		}
		std::string vstring = oss.str();

		return vstring;
	}

	std::string Alg_Foundation::Impl::vectors2string_string(std::vector<BoxInfo> vectors) {
		if (vectors.size() == 0) { return ""; }

		std::string _tmp;
		for (int i = 0; i < vectors.size(); i++)
		{
			_tmp += labelNames[vectors[i].label];
			if (vectors.size() != i + 1)
			{
				_tmp += "-";
			}
		}

		return _tmp;
	}

	std::string Alg_Foundation::Impl::vectors2string_string(std::vector<int> vectors) {
		if (vectors.size() == 0) { return ""; }

		std::string _tmp;
		for (int i = 0; i < vectors.size(); i++)
		{
			_tmp += labelNames[vectors[i]];
			if (vectors.size() != i + 1)
			{
				_tmp += "-";
			}
		}

		return _tmp;
	}

	void Alg_Foundation::Impl::updateparamfromdlg(CheckParam _param)
	{
		m_checkparam = _param;
	}

	double Alg_Foundation::Impl::calBoxArea(BoxInfo boxinfo)
	{
		double boxWidth = std::abs(boxinfo.x2 - boxinfo.x1);
		double boxHeight = std::abs(boxinfo.y2 - boxinfo.y1);

		double boxArea = boxWidth * boxHeight;
		return boxArea;
	}
	
	double Alg_Foundation::Impl::calBoxDiagonalLength(BoxInfo boxinfo)
	{
		double boxWidth = std::abs(boxinfo.x2 - boxinfo.x1);
		double boxHeight = std::abs(boxinfo.y2 - boxinfo.y1);

		double boxDiagonalLength = std::sqrt(boxWidth * boxWidth + boxHeight * boxHeight);

		return boxDiagonalLength;
	}

	bool Alg_Foundation::Impl::initResponseAndVisual()
	{
		result = "FAIL";;  // response relative vars
		ocr_content = "";
		cell_num = "";
		snfrom = "";
		ERCODE = "";
		ERMSG = "";

		mark_color = colorMap["red"];
		error_res.clear();

		return 1;
	}

	bool Alg_Foundation::Impl::generateResponseAndVisual(const std::vector<int>& error_res, const std::string& ocr_content, wikky_algo::SingleMat & data)
	{

		LOGT("Total error size:   {}; \n{};", error_res.size(), vectors2string(error_res));
		LOGT("Total error string: {};", vectors2string_string(error_res));



		// error_res 去重处理
		std::vector<int> uniqueVec;
		std::set<int> uniqueSet(error_res.begin(), error_res.end());
		std::copy(uniqueSet.begin(), uniqueSet.end(), std::back_inserter(uniqueVec));

		

		// response 变量处理
		result = (error_res.size() != 0) ? "FAIL" : "PASS";
		bool isNull = (std::find(error_res.begin(), error_res.end(), error_code_length-2) != error_res.end());
		if (isNull) { result = "NULL"; }

		QStringList tmp = QString::fromStdString(data.sn_fromscanner).split("-");
		cell_num = tmp.size() == 2 ? tmp[1].toStdString() : "";
		snfrom   = tmp.size() == 2 ? tmp[0].toStdString() : "";

		this->ocr_content = ocr_content;

		ERCODE = vectors2string(uniqueVec);
		ERMSG = vectors2string_string(uniqueVec);



		// visual 使用
		mark_color = (isNull) ? colorMap["null"] : (result == "PASS") ? colorMap["green"] : colorMap["red"];

		cv::putText(data.imgrst, (isNull)? "NU": (result == "PASS") ? "OK" : "NG", cv::Point(20, 300), 3, 10, mark_color, 30);
		
		std::string _tmp = (result == "PASS") ? "OK" : ERMSG;
		for (size_t i = 0; i < uniqueVec.size(); i++)
		{
			cv::putText(data.imgrst, "R: " + labelNames[uniqueVec[i]], cv::Point(120, 1400 + 70 * i), 1, 5, mark_color, 10);
		}
		LOGW("==> Result({}): {};", error_res.size(), (result == "PASS") ? "OK" : ERMSG);



		// error_message 使用
		data.error_message.push_back(result);
		data.error_message.push_back(snfrom);
		data.error_message.push_back(this->ocr_content);
		data.error_message.push_back(cell_num);
		data.error_message.push_back(ERCODE);
		data.error_message.push_back(ERMSG);



		LOGW("==> {};", vectors2string(data.error_message));
		return 1;
	}

	Alg_Foundation::Impl::Impl()
	{
		LOGSET(SPDLOG_LEVEL_TRACE);
		LOGT("DLLInterface:{}", DLLINTERFACE);
		tid = std::this_thread::get_id();
		t = *(_Thrd_t*)(char*)&tid;
		unsigned int nId = t._Id;
		itoa(nId, buf, 10);

		SetConsoleOutputCP(CP_UTF8);

		wikky_algo::ColorManager colorManager = wikky_algo::ColorManager();
		colorManager.initColorMap();
		colorMap = colorManager.colorMap;

	}

	Alg_Foundation::Impl::~Impl()
	{
		saveAlgoParam();
	}
	
	bool Alg_Foundation::Impl::initAlgoparam(std::string& _camserial)
	{
		try {

			//! 初始化模型执行所需要的参数
			// 1 模型相关文件的定义和赋值
			string model_path_list = qApp->applicationDirPath().toStdString()    + "/model_files";
			std::string onnxModelPath = qApp->applicationDirPath().toStdString() + "/model_files/3_6_best.onnx";
			std::string labelPath = qApp->applicationDirPath().toStdString()     + "/model_files/3_6_classnames.txt";


			// 2 获取YAML参数数据；便于在构造函数中对某些config变量进行赋值
			m_scamserial = _camserial;
			readAlgoParam();

			readLabelNames(labelPath, labelNames);
			labelNames.push_back("NG_NULL");
			labelNames.push_back("NG_UKOWN");
			error_code_length = labelNames.size();


			for (int i = 0; i < labelNames.size(); i++)
			{
				LOGT("[read label]: {}->{};", i, labelNames[i]);
			}


			// 3 初始化模型执行所需要的参数集合
			config.modelpath = onnxModelPath;
			config.confThreshold = m_checkparam.th_conf;
			config.nmsThreshold = m_checkparam.th_nms;
			config.objThreshold = m_checkparam.th_obj;
			config.ironcladScratchThreshold   = m_checkparam.ironcladScratchThreshold;
			config.ironcladShinyMarkThreshold = m_checkparam.ironcladShinyMarkThreshold;
			config.ironcladResiGlueThreshold  = m_checkparam.ironcladResiGlueThreshold;
			config.ironcladDentThreshold      = m_checkparam.ironcladDentThreshold;
			config.bootScratchThreshold       = m_checkparam.bootScratchThreshold;
			config.bootDentThreshold          = m_checkparam.bootDentThreshold;
			config.contaminationThreshold     = m_checkparam.contaminationThreshold;

			LOGT("[Labels_with_length init]: {};", vectors2string(labels_with_length));


			//! 实例化模型对象
			bool isGPU = true;
			firedetmodel = new YOLOv5(config);

			//! 先跑一次
			LOGW("Start: the first model interface.");
			cv::Mat m = cv::imread("D:/1.bmp", 1);
			frames.push_back(m);
			firedetmodel->detect(frames, output);
			frames.clear();
			output.clear();
			LOGW("End:   the first model interface.");

		}
		catch (std::exception& e) {
			LOGW("s: {}", e.what());
			return false;
		}

		return true;
	}

	bool Alg_Foundation::Impl::popCameraDlg(void* parent)
	{
		if (nullptr == algosettingdlg)
		{
			algosettingdlg = std::make_shared<Qtalgosettingdlg>((QWidget*)parent);
			algosettingdlg->SetTestCallback(std::bind(&Alg_Foundation::Impl::doing, this, std::placeholders::_1, std::placeholders::_2));
			algosettingdlg->UpdatetoalgoImpl(std::bind(&Alg_Foundation::Impl::updateparamfromdlg, this, std::placeholders::_1));
		}

		algosettingdlg->SetLastImage(lastimg);
		Param2Node(m_checkparam, m_yamlparams);
		algosettingdlg->SetLastParam(YAML::Clone(m_yamlparams));  //! 重大BUG修复之处
		algosettingdlg.get()->show();
		LOGW("popCameraDlg successfully");
		return false;
	}

	bool Alg_Foundation::Impl::readAlgoParam()
	{
		//QSettings algsetting(qApp->applicationDirPath() + "/defaultModel/" + m_scamserial.c_str() + ".ini", QSettings::IniFormat);
		//m_checkparam._iThread = algsetting.value("Default1/_Thread", 100).toInt();


		std::string _scamserial = m_scamserial;
		std::replace(_scamserial.begin(), _scamserial.end(), '/', '_');
		std::replace(_scamserial.begin(), _scamserial.end(), ':', '_');

		QString str = QString("%1/defaultModel/%2.yaml").arg(qApp->applicationDirPath()).arg(_scamserial.c_str());
		try
		{
			m_yamlparams = YAML::LoadFile(str.toStdString());
			Node2Param(m_checkparam, m_yamlparams);

			std::filesystem::path filepath = std::filesystem::path(str.toLocal8Bit().constData());
			LOGW("load [{}] successfully.", filepath.filename().string());
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			LOGW("load {} error: {}", str.toStdString(), e.what());
			return false;
		}

		return true;
	}

	bool Alg_Foundation::Impl::saveAlgoParam()
	{
		//QSettings algsetting(qApp->applicationDirPath() + "/defaultModel/" + m_scamserial.c_str() + ".ini", QSettings::IniFormat);
		//algsetting.setValue("Default1/_Thread", m_checkparam._iThread);

		std::string _scamserial = m_scamserial;
		std::replace(_scamserial.begin(), _scamserial.end(), '/', '_');
		std::replace(_scamserial.begin(), _scamserial.end(), ':', '_');

		QString str = QString("%1/defaultModel/%2.yaml").arg(qApp->applicationDirPath()).arg(_scamserial.c_str());
		QDir targetDir(str);
		targetDir.remove(str);

		std::ofstream fout;

		fout.open(str.toStdString().c_str(), ios::out | ios::trunc);
		Param2Node(m_checkparam, m_yamlparams);

		try
		{
			fout << m_yamlparams;
			fout.flush();
			fout.close();
		}
		catch (YAML::ParserException e)
		{
		}
		catch (YAML::RepresentationException e)
		{
		}
		catch (YAML::Exception e)
		{
		}
		return false;
	}
	
	bool Alg_Foundation::Impl::setLogLevel(int _i)
	{
		LOGSET(_i);
		return true;
	}

	bool Alg_Foundation::Impl::readLabelNames(std::string& label_file_path, std::vector<std::string>& labelNames) {
		std::ifstream infile(label_file_path);     // 打开文件

		if (!infile) {
			LOGE("无法打开Label文件");
			return false;
		}

		std::string line;
		while (std::getline(infile, line)) {      // 逐行读取文件内容
			std::istringstream iss(line);         // 使用字符串流处理每行内容

			std::string word;
			while (iss >> word) {                 // 从字符串流中提取单词
				labelNames.push_back(word);       // 将单词添加到向量中
			}
		}

		infile.close();                           // 关闭文件

		return true;
	}

	int Alg_Foundation::Impl::doing(wikky_algo::SingleMat& data, wikky_algo::CheckParam* _checkparam)
	{
		LOGT("\n");

		//! 1 preprocess iamge and init vars
		cv::Mat bgr_img;
		if (data.imgori.channels() == 1) {
			cv::cvtColor(data.imgori, bgr_img, cv::COLOR_GRAY2BGR);
		}
		else {
			bgr_img = data.imgori.clone();
		}
		
		data.imgrst = bgr_img.clone();
		lastimg = data.imgori.clone();
		cv::Mat ma = bgr_img.clone();

		if ("local_camera" == m_scamserial) {
			LOGW("local mode. {};", m_scamserial);
			data.sn_fromscanner = "local-camera";
		}


		// 我们需要BGR图像，但是相机给的是灰度图像

		// 初始化 visual 和 response 相关变量
		initResponseAndVisual();

		try {

			//! 空板判断-开始
			cv::Rect kong_roi = cv::Rect(cv::Point(1236, 600), cv::Point(1650, 1360));
			cv::Mat gray_m;
			cv::cvtColor(ma, gray_m, cv::COLOR_RGB2GRAY);
			cv::Mat bi_gray_m;
			cv::threshold(gray_m, bi_gray_m, 110, 255, cv::THRESH_BINARY);
			cv::Mat kong_image = bi_gray_m.clone();
			cv::Mat kong_roi_img = kong_image(kong_roi);

#ifdef _DEBUG
			cv::rectangle(data.imgrst, kong_roi, colorMap["null"], 5);
#endif // DEBUG


			int non_zero_num = cv::countNonZero(kong_roi_img);
			LOGW("non_zero_num: {};", non_zero_num);

			if (non_zero_num < 100)
			{
				LOGW("there is no product  .");
				error_res.push_back(error_code_length - 2);
				generateResponseAndVisual(error_res, "", data);

				return -1;
			}
			//! 空板判断结束
			
			//else
			{
				//! 模型运行
				frames.push_back(ma);
				output.clear();
				bool return_bool = firedetmodel->detect(frames, output);
				frames.clear();



				//! 模型输出结果整理
				if (return_bool == false) {
					result = "FAIL";
					error_res.push_back(error_code_length - 1);
				}
				else {
					cv::Scalar _tmp_color;

					//for (size_t i = 0; i < output.size(); ++i) 
					{
						for (size_t j = 0; j < output[0].size(); ++j)
						{
							BoxInfo cur_box = output[0][j];

							cv::Point _tmp_pt1 = cv::Point(int(cur_box.x1), int(cur_box.y1));
							cv::Point _tmp_pt2 = cv::Point(int(cur_box.x2), int(cur_box.y2));


							// size filt start
							double box_length_val = calBoxDiagonalLength(cur_box);
							double box_area_val = calBoxArea(cur_box);


							double standard_val = -1.0;
							double box_val = -1.0;
							if ("ironcladScratch" == labelNames[cur_box.label])        { standard_val = config.ironcladScratchThreshold;   box_val = box_length_val; }
							else if ("ironcladShinyMark" == labelNames[cur_box.label]) { standard_val = config.ironcladShinyMarkThreshold; box_val = box_area_val; }
							else if ("ironcladResiGlue" == labelNames[cur_box.label])  { standard_val = config.ironcladResiGlueThreshold;  box_val = box_area_val; }
							else if ("ironcladDent" == labelNames[cur_box.label])      { standard_val = config.ironcladDentThreshold;      box_val = box_area_val; }
							else if ("bootScratch" == labelNames[cur_box.label])       { standard_val = config.bootScratchThreshold;       box_val = box_length_val; }
							else if ("bootDent" == labelNames[cur_box.label])          { standard_val = config.bootDentThreshold;          box_val = box_area_val; }
							else if ("contamination" == labelNames[cur_box.label])     { standard_val = config.contaminationThreshold;     box_val = box_area_val; }
							else { LOGW("unkown label"); standard_val = config.contaminationThreshold;     box_val = box_area_val; }  // 默认


							if (box_val < standard_val)
							{
								_tmp_color = colorMap["green"];
							}
							else {
								_tmp_color = colorMap["red"];
								error_res.push_back(cur_box.label);
							}


							cv::rectangle(data.imgrst, _tmp_pt1, _tmp_pt2, _tmp_color, 3);

							cv::Point _tmp_pt3 = cv::Point(int(output[0][j].x1), int(output[0][j].y1) - 5);
							cv::putText(data.imgrst, "label: " + labelNames[cur_box.label], _tmp_pt3, cv::FONT_HERSHEY_SIMPLEX, 0.75, _tmp_color, 1);

							cv::Point _tmp_pt4 = cv::Point(int(output[0][j].x1), int(output[0][j].y1) - 30);
							cv::putText(data.imgrst, "score: " + cv::format("%.2f", cur_box.score), _tmp_pt4, cv::FONT_HERSHEY_SIMPLEX, 0.75, _tmp_color, 1);
						}
					}
				}
				generateResponseAndVisual(error_res, "", data);
			}

		}
		catch (std::exception& e) {
			LOGW("s: {}", e.what());
		}

		
		//! 5 general response block；
		_iTest++;
		ERCODE = std::to_string(_iTest);

		return -1;
	}

};
