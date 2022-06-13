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

  void wait_default_stream() {
    cudaStreamSynchronize(0);
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

  void check_and_sync() {
	  CHECK_LAST_KERN();
  }

  uint32_t get_device_count() {
	  int count;
	  cudaGetDeviceCount(&count);
	  return (uint32_t)count;
  }

  int32_t get_device_cuda_core_count() {
	  cudaDeviceProp devProp;
      cudaGetDeviceProperties(&devProp, 0);

	  int cores = -1;
	  int mp = devProp.multiProcessorCount;

	  switch (devProp.major){
		  case 2: // Fermi
			  if (devProp.minor == 1) cores = mp * 48;
			  else cores = mp * 32;
			  break;
		  case 3: // Kepler
			  cores = mp * 192;
			  break;
		  case 5: // Maxwell
			  cores = mp * 128;
			  break;
		  case 6: // Pascal
			  if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
			  else if (devProp.minor == 0) cores = mp * 64;
			  break;
		  case 7: // Volta and Turing
			  if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
			  break;
		  case 8: // Ampere
			  if (devProp.minor == 0) cores = mp * 64;
			  else if (devProp.minor == 6) cores = mp * 128;
			  break;
		  default:
			  break;
	  }
	  return cores;
  }

  void cu_memset(uint8_t* devPtr, uint8_t value, size_t count) {
    CHECK(cudaMemset((void*)devPtr, value, count));
  }
}

