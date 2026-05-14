#include <cuda_runtime.h>
#include "moonai_gpu_ffi.hpp"

extern "C" uint32_t c_cuda_malloc(void **ptr, std::size_t size) {
  return cudaMalloc(ptr, size);
}

extern "C" uint32_t c_cuda_free(void *ptr) {
  if (ptr == nullptr) {
    return 0;
  }

  return cudaFree(ptr);
}

extern "C" uint32_t c_cuda_memset_zero(void *ptr, std::size_t size) {
  if (ptr == nullptr || size == 0U) {
    return 0;
  }

  return cudaMemset(ptr, 0, size);
}

extern "C" uint32_t c_cuda_host_to_dev(void *dev_ptr, const void *host_ptr, std::size_t size) {
  if (size == 0U) {
    return 0;
  }

  return cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice);
}

extern "C" uint32_t c_cuda_dev_to_dev(void *dst_ptr, const void *src_ptr, std::size_t size) {
  if (size == 0U) {
    return 0;
  }

  return cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice);
}
