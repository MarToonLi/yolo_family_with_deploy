include(FetchContent)
if(NOT WIKKYDLL_Found)

FetchContent_Declare(
  WIKKYDLL
  GIT_REPOSITORY    git@github.com:wikky250/wikky_dll.git
  GIT_TAG           master
)

FetchContent_MakeAvailable(WIKKYDLL)
endif()