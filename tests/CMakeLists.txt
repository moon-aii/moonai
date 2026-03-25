find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(moonai_tests
    test_genome.cpp
    test_neural_network.cpp
    test_config.cpp
    test_random.cpp
    test_logger.cpp
    test_evolution.cpp
    test_ecs_entity.cpp
    test_ecs_sparse_set.cpp
    test_ecs_registry.cpp
    test_ecs_performance.cpp
    test_ecs_systems.cpp
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

# GPU tests — only built when CUDA is available
if(MOONAI_ENABLE_CUDA)
    add_executable(moonai_gpu_tests
        test_gpu.cpp
        test_ecs_gpu_packing.cpp
    )

    target_link_libraries(moonai_gpu_tests PRIVATE
        GTest::gtest
        GTest::gtest_main
        moonai_core
        moonai_evolution
        moonai_gpu
    )

    target_compile_definitions(moonai_gpu_tests PRIVATE MOONAI_ENABLE_CUDA)

    gtest_discover_tests(moonai_gpu_tests)
endif()
