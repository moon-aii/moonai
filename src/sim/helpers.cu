#include <cuda_runtime.h>
#include "moonai_gpu_ffi.hpp"

extern "C" uint32_t c_cuda_malloc(void **ptr, std::size_t size) {
  return cudaMalloc(ptr, size);
}

extern "C" uint32_t c_cuda_free(void **ptr) {
  return cudaFree(ptr);
}

extern "C" uint32_t c_cuda_memset_zero(void *ptr, std::size_t size) {
  return cudaMemset(ptr, 0, size);
}

extern "C" uint32_t c_cuda_dev_to_host(const void *dev_ptr, void *host_ptr, std::size_t size) {
  return cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost);
}

extern "C" uint32_t c_cuda_host_to_dev(void *dev_ptr, const void *host_ptr, std::size_t size) {
  return cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice);
}

extern "C" uint32_t c_cuda_sync() { 
  return cudaDeviceSynchronize();
}
