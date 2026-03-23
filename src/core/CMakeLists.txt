add_library(moonai_core STATIC
    config.cpp
    random.cpp
)

target_include_directories(moonai_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_core PUBLIC
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    sol2::sol2
    ${LUA_LIBRARIES}
)

target_include_directories(moonai_core PUBLIC ${LUA_INCLUDE_DIR})
