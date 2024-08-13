#pragma once

#include <condition_variable>
#include <memory>
#include <string>
#include <list>
#include <stdint.h>
#include <thread>
#include <chrono>
#include <future>
#include <functional>
#include <intsafe.h>
#include "common.h"

namespace wikky_algo
{
class IBaseAlg
{
public:
    virtual ~IBaseAlg() {}
    virtual bool initAlgoparam(std::string&) = 0;
    virtual bool popCameraDlg(void* parent) = 0;
    virtual bool setLogLevel(int) = 0;
    virtual int doing(wikky_algo::SingleMat&, wikky_algo::CheckParam* m_checkparam = nullptr) = 0;
};

typedef IBaseAlg* (__stdcall* pExportALG)();
typedef void (__stdcall* pDeleteALG)(IBaseAlg*);
}  // namespace cameramanager