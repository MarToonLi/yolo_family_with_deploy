
find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
add_library(qmytreewidget INTERFACE)

target_sources(qmytreewidget
INTERFACE
    qmytreewidget.h
    qmytreewidget.cpp
    140.png
    240.png
    340.png
    440.png
    540.png
    btn.png
)
target_link_libraries(qmytreewidget
INTERFACE
    logging
    common
    Qt5::Widgets
    yaml-cpp
)

target_include_directories(qmytreewidget 
INTERFACE
	./
)