set(MOONAI_GPU_SOURCES
    gpu_data_buffer.cu
    gpu_entity_mapping.cpp
    ecs_gpu_packing.cu
    gpu_batch.cu
    gpu_network_cache.cu
)

add_library(moonai_gpu STATIC ${MOONAI_GPU_SOURCES})
add_library(moonai_gpu_profiler STATIC ${MOONAI_GPU_SOURCES})

target_include_directories(moonai_gpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_include_directories(moonai_gpu_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)

# Compile for the GPU installed on the build machine
set_property(TARGET moonai_gpu PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET moonai_gpu_profiler PROPERTY CUDA_ARCHITECTURES native)

# No CUDA_SEPARABLE_COMPILATION: our kernels have no cross-TU __device__ calls,
# so each .cu is self-contained. Separable compilation adds device-link complexity
# with static libraries and is not needed here.

target_link_libraries(moonai_gpu PUBLIC
    moonai_core
)

target_link_libraries(moonai_gpu_profiler PUBLIC
    moonai_core_profiler
)

target_compile_definitions(moonai_gpu_profiler PUBLIC MOONAI_BUILD_PROFILER)
