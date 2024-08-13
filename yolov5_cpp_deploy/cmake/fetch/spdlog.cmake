include(FetchContent)

if(THIRDPARTY_DIR)
FetchContent_Declare(spdlog
    GIT_REPOSITORY ${THIRDPARTY_DIR}/spdlog
    GIT_TAG v1.11.0
)
else()
FetchContent_Declare(spdlog
    GIT_REPOSITORY git@github.com:gabime/spdlog.git
    GIT_TAG v1.11.0
)
endif()

FetchContent_MakeAvailable(spdlog)