find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(moonai_tests
    test_evolution.cpp
    test_config.cpp
    test_food_store.cpp
    test_random.cpp
    test_logger.cpp
    test_simulation_sensors.cpp
)

target_link_libraries(moonai_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    moonai_core
    moonai_simulation
    moonai_evolution
    moonai_data
)

gtest_discover_tests(moonai_tests)
