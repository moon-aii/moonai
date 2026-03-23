add_library(moonai_gpu STATIC
    neural_inference.cu
    gpu_batch.cu
)

target_include_directories(moonai_gpu PUBLIC ${CMAKE_SOURCE_DIR}/src)

# Compile for the GPU installed on the build machine
set_property(TARGET moonai_gpu PROPERTY CUDA_ARCHITECTURES native)

# No CUDA_SEPARABLE_COMPILATION: our kernels have no cross-TU __device__ calls,
# so each .cu is self-contained. Separable compilation adds device-link complexity
# with static libraries and is not needed here.

target_link_libraries(moonai_gpu PUBLIC
    moonai_core
)
