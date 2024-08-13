include(FetchContent)

if(THIRDPARTY_DIR)
message(STATUS "concurrentqueue src dir : ${THIRDPARTY_DIR}/concurrentqueue")
FetchContent_Declare(concurrentqueue
    GIT_REPOSITORY ${THIRDPARTY_DIR}/concurrentqueue
)
else()
FetchContent_Declare(
  concurrentqueue
  GIT_REPOSITORY    git@github.com:cameron314/concurrentqueue.git
  GIT_TAG           master
)
endif()

FetchContent_MakeAvailable(concurrentqueue)