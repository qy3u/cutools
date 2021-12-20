#include <cstdint>
#include <stdio.h>

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

#define CHECK_LAST_KERN()\
{\
  cudaDeviceSynchronize();\
  const cudaError_t error=cudaGetLastError();\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}


struct StreamWraper {
  cudaStream_t stream;

  StreamWraper() {
    CHECK( cudaStreamCreate(&stream) );
  }

  ~StreamWraper() {
    CHECK( cudaStreamDestroy(this->stream) );
  }

  cudaStream_t get() {
    return this->stream;
  }
};


extern "C" {
  void* create_stream() {
    StreamWraper* s = new StreamWraper();
    return s;
  }

  void destory_stream(void* stream) {
    StreamWraper* _stream = (StreamWraper*)stream;
    delete _stream;
  }

  void wait_stream(void* stream) {
    StreamWraper* s = (StreamWraper*)stream;
    cudaStreamSynchronize(s->get());
  }

  void* get_inner_stream(void* stream) {
    StreamWraper* s = (StreamWraper*)stream;
	return (void*)&s->stream;
  }
  
  uint8_t* alloc_locked_buffer(uint32_t bytes) {
    uint8_t* buffer;

    CHECK(cudaMallocHost(&buffer, bytes));

    return buffer;
  }

  void free_locked_buffer(uint8_t* buffer) {
    CHECK(cudaFreeHost(buffer));
  }

  void set_device(size_t index) {
    CHECK(cudaSetDevice(index));
  }

  void device_to_host(uint8_t* device, uint8_t* host, uint32_t bytes) {
    CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
  }

  void device_to_host_with_stream(uint8_t* device, uint8_t* host, uint32_t bytes, void* stream) {
    StreamWraper* s = (StreamWraper*)stream;
    CHECK( cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, s->get()) );
  }


  void host_to_device(uint8_t* host, uint8_t* device, uint32_t bytes) {
    CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
  }

  void host_to_device_with_stream(uint8_t* host, uint8_t* device, uint32_t bytes, void* stream) {
    StreamWraper* s = (StreamWraper*)stream;
    CHECK( cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, s->get()) );
  }

  uint8_t* alloc_gpu_buffer(uint32_t bytes) {
    uint8_t* buf;
    CHECK(cudaMalloc(&buf, bytes));
    return buf;
  }

  void free_gpu_buffer(uint8_t* buf) {
    CHECK(cudaFree(buf));
  }
}

