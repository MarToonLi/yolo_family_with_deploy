
find_package(Qt5 COMPONENTS Core REQUIRED)
add_library(3_6_Algo SHARED "")

target_sources(3_6_Algo
PUBLIC 
    foundation.h
    implfoundation.h
    yolov5.hpp
PRIVATE 
    foundation.cpp
    implfoundation.cpp
    yolov5.cpp
)

# 该.h路径块需要防止到public下，如果是private下，会出现引用问题，但不影响编译和执行。
target_include_directories(3_6_Algo
PUBLIC
    ./
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/include/onnxruntime/core/session  
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/include
PRIVATE
    ./
)



target_link_directories(3_6_Algo
PUBLIC
    ./
)

target_link_libraries(3_6_Algo
PUBLIC 
    algosettingdlg
    logging
    common
    base_alg
    Qt5::Core
    yaml-cpp
PRIVATE
    algosettingdlg
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/build/Windows/Release/Release/onnxruntime.lib
)