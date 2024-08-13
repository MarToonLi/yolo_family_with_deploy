find_package(OpenCV)

add_library(common INTERFACE)

target_sources(common INTERFACE common.h common.cpp)

target_include_directories(common INTERFACE ./)

target_link_libraries(common
INTERFACE
	opencv_world
)