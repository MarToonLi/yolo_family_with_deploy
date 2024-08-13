#ifndef CORE_COMMON
#define CORE_COMMON
#include <opencv2/opencv.hpp>
namespace wikky_algo
{
#define DLLINTERFACE "1.1"
	struct SingleMat
	{
		int camPos = -1;
		std::chrono::steady_clock::time_point starttime;
		cv::Mat imgori;
		cv::Mat imgrst;
		size_t index;
		int groupsize;
		bool bresult;
		std::string sn_fromscanner;
		std::string cam_serial;
		std::vector<std::string> error_message;
	};
	struct CheckParam
	{
		std::string cam_serial;
		float th_conf = 0.7;
		float th_nms = 0.45;
		float th_obj = 0.7;

		float ironcladScratchThreshold   = 30;
		float ironcladShinyMarkThreshold = 900;
		float ironcladResiGlueThreshold  = 900;
		float ironcladDentThreshold      = 900;
		float bootScratchThreshold       = 30;
		float bootDentThreshold          = 900;
		float contaminationThreshold     = 900;

	};

	class ColorManager {
	public:
		std::map<std::string, cv::Scalar> colorMap;

		void initColorMap();
	};

	using TestCallback = std::function<int(SingleMat&, CheckParam*)>;
	using UpdateParam = std::function<void(CheckParam&)>;
}

#endif // !CORE_COMMON