cmake_minimum_required(VERSION 3.15)
project(Compass_Project VERSION 0.1.0 LANGUAGES CXX)
find_package(Qt5 COMPONENTS Core Gui Widgets Network Sql REQUIRED)

if(MSVC)
add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING )
endif()
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(fetch/yamlcpp)
include(fetch/spdlog)
include(fetch/concurrentqueue)

message(STATUS "THIRDPARTY_DIR src dir : ${THIRDPARTY_DIR}")
#set (WIKKYDLL_DIR "D:/MyCode/new_compose/AlgDll_Wikky")
#message(STATUS "WIKKYDLL_DIR src dir : ${WIKKYDLL_DIR}")

if(DEFINED THIRDPARTY_DIR)
    set(OpenCV_DIR ${THIRDPARTY_DIR}/opencv455_DNN/install/x64/vc17/lib)

endif()



set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG  ${PROJECT_SOURCE_DIR}/bin/debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release)

add_compile_options(
    $<$<AND:$<NOT:$<COMPILE_LANGUAGE:CUDA>>,$<OR:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>>:/utf-8>
)

add_subdirectory(src)
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG  ${PROJECT_SOURCE_DIR}/bin/debug)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release)
    add_subdirectory(example-demo)
endif()