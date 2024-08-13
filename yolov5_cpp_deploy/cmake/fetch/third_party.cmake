FetchContent_Declare(THIRDPARTY
    URL ${THIRDPARTY_URL}
    URL_MD5 ${THIRDPARTY_MD5}
    # SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party
)

message(STATUS "Fetch THIRDPARTY")
FetchContent_MakeAvailable(THIRDPARTY)
message(STATUS "Fetch THIRDPARTY done")

FetchContent_GetProperties(THIRDPARTY SOURCE_DIR THIRDPARTY_DIR)
