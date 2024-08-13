include(FetchContent)
if(THIRDPARTY_DIR)
FetchContent_Declare(yamlcpp
    GIT_REPOSITORY ${THIRDPARTY_DIR}/yaml-cpp
)
else()
FetchContent_Declare(
  yamlcpp
  GIT_REPOSITORY    git@github.com:jbeder/yaml-cpp.git
  GIT_TAG           master
)
endif()

FetchContent_MakeAvailable(yamlcpp)