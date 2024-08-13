#include "ibasealgorithm.h"
namespace wikky_algo
{
class Alg_Foundation : public wikky_algo::IBaseAlg
{
   public:
       Alg_Foundation();
       ~Alg_Foundation();


       bool initAlgoparam(std::string&);

       bool popCameraDlg(void* parent);

       bool setLogLevel(int);

       int doing(wikky_algo::SingleMat&, wikky_algo::CheckParam* m_checkparam = nullptr);


   private:
       class Impl;
       std::shared_ptr<Impl> impl_;

};
    
}  // namespace wikky_algo
extern "C" __declspec(dllexport) wikky_algo::IBaseAlg * __stdcall CreateExportAlgObj();
extern "C" __declspec(dllexport) void __stdcall DeleteExportAlgObj(
    wikky_algo::IBaseAlg*);