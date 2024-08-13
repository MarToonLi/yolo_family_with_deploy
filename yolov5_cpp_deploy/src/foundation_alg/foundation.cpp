#include "foundation.h"
#include "implfoundation.h"

#include <iostream>
#include "logger.h"
namespace wikky_algo
{

    Alg_Foundation::Alg_Foundation() : impl_(std::make_unique<Impl>()) {}

    Alg_Foundation::~Alg_Foundation() {}

    bool Alg_Foundation::initAlgoparam(std::string& camserial)
    {
        return impl_->initAlgoparam(camserial);
    }

    bool Alg_Foundation::popCameraDlg(void* parent)
    {
        return impl_->popCameraDlg(parent);
    }

    bool Alg_Foundation::setLogLevel(int _i)
    {
        return impl_->setLogLevel(_i);
    }
    int Alg_Foundation::doing(wikky_algo::SingleMat& data, wikky_algo::CheckParam* m_checkparam)
    {
        return impl_->doing(data, m_checkparam);
    }

}

extern "C" __declspec(dllexport) wikky_algo::IBaseAlg * __stdcall CreateExportAlgObj()
{
    return new wikky_algo::Alg_Foundation();
};
extern "C" __declspec(dllexport) void __stdcall DeleteExportAlgObj(
    wikky_algo::IBaseAlg * p)
{
    delete p;
    p = nullptr;
};