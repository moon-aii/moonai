find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(moonai_tests
    test_evolution.cpp
)

target_link_libraries(moonai_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    moonai_core
    moonai_simulation
    moonai_evolution
)

gtest_discover_tests(moonai_tests)
