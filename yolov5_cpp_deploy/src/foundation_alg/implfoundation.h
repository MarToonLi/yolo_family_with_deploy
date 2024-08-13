#include "ibasealgorithm.h"
#include "algosettingdlg.h"
#include "yaml-cpp/yaml.h"
#include "yolov5.hpp"

namespace wikky_algo
{
    std::string d2str(double val, int precision = 3);

    std::string vectors2string(std::vector<std::string> vectors);

    std::string vectors2string(std::vector<int> vectors);

    std::string vectors2string(std::vector<BoxInfo> vectors);

    class Alg_Foundation::Impl : public wikky_algo::IBaseAlg
    {
    private:
        std::thread::id tid;
        _Thrd_t t;
        char* buf = new char[10];
        cv::Mat lastimg;
        std::shared_ptr<Qtalgosettingdlg> algosettingdlg = nullptr;
        std::string m_scamserial;

        double SMALL_AREA_THRESHOLD = 2.3e+06;
        double LARGE_AREA_THRESHOLD = 2.7e+06;

        // checkparam
        CheckParam m_checkparam;
        YAML::Node m_yamlparams;
        void updateparamfromdlg(CheckParam _param);
    /// <summary>
        std::vector<std::vector<cv::Point>>contours, con_temp, contours_Selected, condidat1, condidat2;
        YOLOv5 *firedetmodel;
        std::vector<cv::Mat> frames;
        std::vector<std::vector<BoxInfo>> output;
        int imageIndex = 0;
        int _iTest = 0;


        //========================================================================================================================
        Configuration config;

        std::vector<std::string> labelNames;
        int error_code_length = -1;

        // colorMaps for imageshow.
        std::map<std::string, cv::Scalar> colorMap;        // 初始化在构造函数中

        std::vector<int> labels_with_length = {0, 4};

        double calBoxArea(BoxInfo boxinfo);
        double calBoxDiagonalLength(BoxInfo boxinfo);
        std::string vectors2string_string(std::vector<BoxInfo> vectors);
        std::string vectors2string_string(std::vector<int> vectors);


        //respose and visualization.
        std::string result = "FAIL";;  // response relative vars
        std::string ocr_content = "";
        std::string snfrom = "";
        std::string cell_num = "";
        std::string ERCODE;
        std::string ERMSG;

        cv::Scalar mark_color;
        std::vector<int> error_res;

        bool initResponseAndVisual();
        bool generateResponseAndVisual(const std::vector<int>& error_res, const std::string& ocr_content, wikky_algo::SingleMat& data);
        //========================================================================================================================

    /// </summary>
    public:
        Impl();
        ~Impl();
        bool initAlgoparam(std::string& camserial);
        bool popCameraDlg(void* parent);
        bool readAlgoParam();
        bool saveAlgoParam();
        bool setLogLevel(int);
        bool readLabelNames(std::string& label_file_path, std::vector<std::string>& labelNames);
        int doing(wikky_algo::SingleMat& data, wikky_algo::CheckParam* m_checkparam = nullptr);

    };
}  // namespace wikky_algo
