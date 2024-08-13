include(FetchContent)

FetchContent_Declare(json
    GIT_REPOSITORY git@github.com:nlohmann/json.git
    GIT_TAG v3.10.4
    GIT_SHALLOW ON
)

FetchContent_MakeAvailable(json)
