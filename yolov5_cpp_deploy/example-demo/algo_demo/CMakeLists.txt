
find_package(OpenCV)
find_package(Qt5 COMPONENTS Core Gui Widgets Network Sql REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

add_executable(AlgoDemo)

target_sources(AlgoDemo
    PRIVATE
    main.cpp
	algo_demo.h
	algo_demo.cpp
	algo_demo.ui
    imagewatch.h
    imagewatch.cpp
)

target_link_libraries(AlgoDemo
PUBLIC
    opencv_world
PRIVATE
	Qt5::Widgets
	Qt5::Core	
    logging
    common
    base_alg
    3_6_Algo
)

target_include_directories(AlgoDemo PUBLIC ./)

add_custom_command(
    TARGET AlgoDemo
    POST_BUILD
    COMMAND "$<TARGET_FILE_DIR:Qt5::qmake>/windeployqt.exe" "--no-translations" $<TARGET_FILE:AlgoDemo>
    WORKING_DIRECTORY $<TARGET_FILE_DIR:AlgoDemo>
)

