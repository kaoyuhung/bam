#include <buffer.h>
#include <ctrl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>
#include <event.h>
#include <fcntl.h>
#include <nvm_admin.h>
#include <nvm_cmd.h>
#include <nvm_ctrl.h>
#include <nvm_error.h>
#include <nvm_io.h>
#include <nvm_parallel_queue.h>
#include <nvm_queue.h>
#include <nvm_types.h>
#include <nvm_util.h>
#include <page_cache.h>
#include <queue.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <unistd.h>
#include <util.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "settings.h"
#include "util.h"
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

#define DIRECTIO_MAX_RW_COUNT ((1ULL << 31) - 4096)
typedef enum {
   GPUMEM = 0,
   UVM_READONLY = 1,
   UVM_DIRECT = 2,
   NV_GDS = 3,
   BAFS_DIRECT = 6,
} mem_type;

using error = std::runtime_error;
using std::string;

// uint32_t n_ctrls = 1;
const char* const sam_ctrls_paths[] = {
    "/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm4", "/dev/libnvm9",
    "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm5", "/dev/libnvm6",
    "/dev/libnvm7", "/dev/libnvm8"};
const char* const intel_ctrls_paths[] = {
    "/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3",
    "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7",
    "/dev/libnvm8", "/dev/libnvm9"};

__global__ void bam_sequential_access_kernel(array_d_t<uint64_t>* dr,
                                             uint64_t n_threads,
                                             unsigned long long* req_count,
                                             uint64_t n_elems) {
   uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
   uint64_t reqs_per_thread = n_elems / n_threads;
   for (int i = 0; i < reqs_per_thread; i++) {
      uint64_t idx = i * n_threads + tid;
      if (idx < n_elems) {
         req_count += (*dr)[idx];
      }
   }
}

__global__ void bam_random_access_kernel(array_d_t<uint64_t>* dr,
                                         uint64_t n_threads,
                                         unsigned long long* req_count,
                                         uint64_t* assignment,
                                         uint64_t reqs_per_thread,
                                         uint64_t n_elems) {
   uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i = 0; i < reqs_per_thread; i++) {
      uint64_t idx = assignment[i * n_threads + tid];
      req_count += (*dr)[idx];
   }
}

__global__ void naive_sequential_access_kernel(uint64_t* arr,
                                               uint64_t n_threads,
                                               uint64_t n_elems) {
   uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
   uint64_t reqs_per_thread = n_elems / n_threads;

   for (int i = 0; i < reqs_per_thread; i++) {
      uint64_t idx = i * n_threads + tid;
      if (idx < n_elems) {
         uint64_t val = arr[idx];
      }
   }
}

__global__ void naive_random_access_kernel(uint64_t* arr,
                                           uint64_t* assignment,
                                           uint64_t n_threads,
                                           uint64_t reqs_per_thread) {
   uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i = 0; i < reqs_per_thread; i++) {
      uint64_t idx = assignment[i * n_threads + tid];
      uint64_t val = arr[idx];
   }
}

static uint64_t round_to_nearest(uint64_t x, uint64_t target) {
   return ((x + target - 1) / target) * target;
}

static size_t get_cpu_free_mem() {
   std::ifstream meminfo("/proc/meminfo");
   std::string line;
   size_t memAvailableKB = 0;
   while (std::getline(meminfo, line)) {
      if (line.find("MemAvailable:") == 0) {
         sscanf(line.c_str(), "MemAvailable: %ld kB", &memAvailableKB);
         break;
      }
   }
   if (memAvailableKB == 0) {
      fprintf(stderr, "Failed to get cpu_free_mem\n");
      exit(EXIT_FAILURE);
   }
   return memAvailableKB * 1024;
}

static inline void print_results(std::string prefix, float time,
                                 uint64_t buff_size) {
   fprintf(stdout, "%s : latency=%.2f ms, throughput=%.2f GB/s\n",
           prefix.c_str(), time, ((double)buff_size / (1 << 30)) / (time / 1000));
}

static float measure_performance(std::function<void(void)> bound_function) {
   //  cudaEvent_t start, stop;
   //  cuda_err_chk(cudaEventCreate(&start));
   //  cuda_err_chk(cudaEventCreate(&stop));
   //  cuda_err_chk(cudaEventRecord(start));

   auto start = std::chrono::high_resolution_clock::now();
   bound_function();
   cudaDeviceSynchronize();
   auto end = std::chrono::high_resolution_clock::now();
   //  cuda_err_chk(cudaEventRecord(stop));
   //  cuda_err_chk(cudaEventSynchronize(stop));
   //  float elapsed_time;
   //  cuda_err_chk(cudaEventElapsedTime(&elapsed_time, start, stop));
   //  cuda_err_chk(cudaEventDestroy(start));
   //  cuda_err_chk(cudaEventDestroy(stop));

   // return elapsed_time;
   return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
       .count();
}

static void GPUMEM_TEST(Settings& settings) {
   if (settings.random) {
      fprintf(stderr, "GPUMEM_TEST does not support random access mode.\n");
      exit(EXIT_FAILURE);
   }

   size_t gpu_free_mem, total_mem;
   cuda_err_chk(cudaMemGetInfo(&gpu_free_mem, &total_mem));
   fprintf(stdout, "gpu_free_mem: %.2lf GiB, total_mem: %.2lf GiB \n",
           (double)gpu_free_mem / (1 << 30), (double)total_mem / (1 << 30));
   uint64_t n_elems = settings.numElems;
   uint64_t data_size = n_elems * sizeof(uint64_t);
   uint64_t buf_size = round_to_nearest(min(data_size, (uint64_t)DIRECTIO_MAX_RW_COUNT), 4096);

   uint64_t n_threads = settings.numThreads;
   uint64_t numblocks = (buf_size / sizeof(uint64_t) + n_threads - 1) / n_threads;
   dim3 blockDim(numblocks);

   fprintf(stdout, "n_elems: %lu, data_size: %lu, buf_size: %lu\n",
           n_elems, data_size, buf_size);
   fprintf(stdout, "numblocks: %lu, n_threads: %lu\n", numblocks, n_threads);

   char *cpu_buf, *dev_buf;
   int fd = open("./test.bin", O_CREAT | O_RDWR | O_DIRECT, 0644);
   if (fd < 0) {
      perror("open");
      close(fd);
      exit(EXIT_FAILURE);
   }
   if (posix_memalign((void**)&cpu_buf, 4096, buf_size) != 0) {
      perror("posix_memalign");
      close(fd);
      std::remove("./test.bin");
      exit(EXIT_FAILURE);
   }
   if (getrandom(cpu_buf, buf_size, 0) == -1) {
      perror("getrandom");
      close(fd);
      std::remove("./test.bin");
      exit(EXIT_FAILURE);
   }
   for (int i = 0; i < (int)((data_size + buf_size - 1) / buf_size); i++) {
      uint64_t bytes2write = min(buf_size, data_size - i * buf_size);
      if (pwrite(fd, cpu_buf, bytes2write, i * buf_size) != (ssize_t)bytes2write) {
         perror("write");
         close(fd);
         std::remove("./test.bin");
         exit(EXIT_FAILURE);
      }
   }
   cuda_err_chk(cudaHostRegister(cpu_buf, buf_size,
                                 cudaHostRegisterDefault));
   cuda_err_chk(cudaMalloc(&dev_buf, buf_size));
   print_results("GPUMEM_TEST",
                 measure_performance([&]() {
                    for (int i = 0; i < (int)((data_size + buf_size - 1) / buf_size); i++) {
                       uint64_t itr_data_size = min(buf_size, data_size - i * buf_size);
                       uint64_t bytes_read = round_to_nearest(itr_data_size, 4096);
                       ssize_t ret;
                       if ((ret = pread(fd, cpu_buf, bytes_read, i * buf_size)) != (ssize_t)bytes_read) {
                          std::cout << "ret: " << ret << ", bytes_read: " << bytes_read << std::endl;
                          perror("pread");
                       }
                       cudaMemcpy(dev_buf, cpu_buf, itr_data_size, cudaMemcpyHostToDevice);
                       naive_sequential_access_kernel<<<blockDim, n_threads>>>(
                           (uint64_t*)dev_buf, numblocks * n_threads, itr_data_size / sizeof(uint64_t));
                    }
                 }),
                 data_size);
   cuda_err_chk(cudaHostUnregister(cpu_buf));
   free(cpu_buf);
   cuda_err_chk(cudaFree(dev_buf));
   std::remove("./test.bin");
}

static void UVM_READONLY_TEST(Settings& settings) {
   size_t cpu_free_mem, gpu_free_mem, total_mem;
   cpu_free_mem = get_cpu_free_mem();
   cuda_err_chk(cudaMemGetInfo(&gpu_free_mem, &total_mem));
   total_mem = (cpu_free_mem + gpu_free_mem) * 0.95;
   fprintf(stdout, "cpu_free_mem: %.2lf GiB, gpu_free_mem: %.2lf GiB, total_mem: %.2lf GiB \n", (double)cpu_free_mem / (1 << 30), (double)gpu_free_mem / (1 << 30), (double)total_mem / (1 << 30));
   uint64_t n_elems = settings.numElems;
   uint64_t data_size = n_elems * sizeof(uint64_t);
   if (settings.random && data_size > total_mem) {
      fprintf(stderr, "Data size exceeds available memory for random access.\n");
      exit(EXIT_FAILURE);
   }
   char* tmp_buf;
   uint64_t tmp_buf_size = round_to_nearest(min(data_size, (uint64_t)DIRECTIO_MAX_RW_COUNT), 4096);
   int fd = open("./test.bin", O_CREAT | O_RDWR | O_DIRECT, 0644);
   if (fd < 0) {
      perror("open");
      close(fd);
      exit(EXIT_FAILURE);
   }
   if (posix_memalign((void**)&tmp_buf, 4096, tmp_buf_size) != 0) {
      perror("posix_memalign");
      close(fd);
      std::remove("./test.bin");
      exit(EXIT_FAILURE);
   }
   if (getrandom(tmp_buf, tmp_buf_size, 0) == -1) {
      perror("getrandom");
      close(fd);
      std::remove("./test.bin");
      exit(EXIT_FAILURE);
   }
   for (int i = 0; i < (int)((data_size + tmp_buf_size - 1) / tmp_buf_size); i++) {
      uint64_t bytes2write = min(tmp_buf_size, data_size - i * tmp_buf_size);
      if (pwrite(fd, tmp_buf, bytes2write, i * tmp_buf_size) != (ssize_t)bytes2write) {
         perror("write");
         close(fd);
         std::remove("./test.bin");
         exit(EXIT_FAILURE);
      }
   }
   free(tmp_buf);
   close(fd);

   uint64_t n_threads = settings.numThreads, buf_size, numblocks, *buf;
   if (settings.random) {
      buf_size = data_size;
      numblocks = (n_threads + 1023) / 1024;
      n_threads = 1024;
   } else {
      buf_size = min(data_size, (uint64_t)1 << 30);
      numblocks = (buf_size / sizeof(uint64_t) + n_threads - 1) / n_threads;
   }
   fprintf(stdout, "n_elems: %lu, data_size: %lu, buf_size: %lu\n",
           n_elems, data_size, buf_size);
   fprintf(stdout, "numblocks: %lu, n_threads: %lu\n", numblocks, n_threads);
   dim3 blockDim(numblocks);
   std::ifstream file("./test.bin", std::ios::in | std::ios::binary);
   if (!file) {
      fprintf(stderr, "Failed to open file: ./test.bin\n");
      exit(EXIT_FAILURE);
   }
   cuda_err_chk(cudaMallocManaged((void**)&buf, buf_size));
   cuda_err_chk(cudaMemAdvise(buf, buf_size, cudaMemAdviseSetReadMostly, settings.cudaDevice));

   if (settings.random) {
      uint64_t *assignment, *d_assignment;
      assignment = (uint64_t*)malloc(settings.numReqs * 1024 * numblocks * sizeof(uint64_t));
      for (uint64_t i = 0; i < settings.numReqs * 1024 * numblocks; i++) {
         assignment[i] = rand() % n_elems;
      }
      cuda_err_chk(cudaMalloc((void**)&d_assignment, settings.numReqs * 1024 * numblocks * sizeof(uint64_t)));
      cuda_err_chk(cudaMemcpy(d_assignment, assignment, settings.numReqs * 1024 * numblocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
      print_results("UVM_READONLY_TEST RANDOM",
                    measure_performance([&]() {
                       file.read((char*)buf, data_size);
                       naive_random_access_kernel<<<blockDim, n_threads>>>(
                           (uint64_t*)buf, d_assignment, numblocks * n_threads,
                           settings.numReqs);
                    }),
                    settings.numReqs * 1024 * numblocks * sizeof(uint64_t));
      free(assignment);
      cuda_err_chk(cudaFree(d_assignment));

   } else {
      print_results("UVM_READONLY_TEST SEQUENTIAL",
                    measure_performance([&]() {
                       for (int i = 0; i < (int)((data_size + buf_size - 1) / buf_size); i++) {
                          uint64_t itr_data_size = min(buf_size, data_size - i * buf_size);
                          file.read((char*)buf, itr_data_size);
                          naive_sequential_access_kernel<<<blockDim, n_threads>>>(
                              (uint64_t*)buf, numblocks * n_threads, itr_data_size / sizeof(uint64_t));
                          cudaDeviceSynchronize();
                       }
                    }),
                    data_size);
   }
   cuda_err_chk(cudaFree(buf));
   file.close();
   std::remove("./test.bin");
}

static void UVM_DIRECT_TEST(Settings& settings) {
   size_t cpu_free_mem, gpu_free_mem, total_mem;
   cpu_free_mem = get_cpu_free_mem();
   cuda_err_chk(cudaMemGetInfo(&gpu_free_mem, &total_mem));
   total_mem = (cpu_free_mem + gpu_free_mem) * 0.95;
   fprintf(stdout, "cpu_free_mem: %.2lf GiB, gpu_free_mem: %.2lf GiB, total_mem: %.2lf GiB \n", (double)cpu_free_mem / (1 << 30), (double)gpu_free_mem / (1 << 30), (double)total_mem / (1 << 30));
   uint64_t n_elems = settings.numElems;
   uint64_t data_size = n_elems * sizeof(uint64_t);
   if (settings.random && data_size > total_mem) {
      fprintf(stderr, "Data size exceeds available memory for random access.\n");
      exit(EXIT_FAILURE);
   }
   char* tmp_buf;
   uint64_t tmp_buf_size = round_to_nearest(min(data_size, (uint64_t)DIRECTIO_MAX_RW_COUNT), 4096);
   int fd = open("./test.bin", O_CREAT | O_RDWR | O_DIRECT, 0644);
   if (fd < 0) {
      perror("open");
      close(fd);
      exit(EXIT_FAILURE);
   }
   if (posix_memalign((void**)&tmp_buf, 4096, tmp_buf_size) != 0) {
      perror("posix_memalign");
      close(fd);
      std::remove("./test.bin");
      exit(EXIT_FAILURE);
   }
   if (getrandom(tmp_buf, tmp_buf_size, 0) == -1) {
      perror("getrandom");
      close(fd);
      std::remove("./test.bin");
      exit(EXIT_FAILURE);
   }
   for (int i = 0; i < (int)((data_size + tmp_buf_size - 1) / tmp_buf_size); i++) {
      uint64_t bytes2write = min(tmp_buf_size, data_size - i * tmp_buf_size);
      if (pwrite(fd, tmp_buf, bytes2write, i * tmp_buf_size) != (ssize_t)bytes2write) {
         perror("write");
         close(fd);
         std::remove("./test.bin");
         exit(EXIT_FAILURE);
      }
   }
   free(tmp_buf);

   uint64_t n_threads = settings.numThreads, buf_size, numblocks, *buf;
   if (settings.random) {
      buf_size = round_to_nearest(data_size, 4096);
      numblocks = (n_threads + 1023) / 1024;
      n_threads = 1024;
   } else {
      buf_size = round_to_nearest(min(data_size, (uint64_t)1 << 30), 4096);
      numblocks = (buf_size / sizeof(uint64_t) + n_threads - 1) / n_threads;
   }
   fprintf(stdout, "n_elems: %lu, data_size: %lu, buf_size: %lu\n",
           n_elems, data_size, buf_size);
   fprintf(stdout, "numblocks: %lu, n_threads: %lu\n", numblocks, n_threads);
   dim3 blockDim(numblocks);
   cuda_err_chk(cudaMallocManaged((void**)&buf, buf_size));
   cuda_err_chk(cudaMemAdvise(buf, buf_size, cudaMemAdviseSetAccessedBy, settings.cudaDevice));
   if (settings.random) {
      uint64_t *assignment, *d_assignment;
      assignment = (uint64_t*)malloc(settings.numReqs * 1024 * numblocks * sizeof(uint64_t));
      for (uint64_t i = 0; i < settings.numReqs * 1024 * numblocks; i++) {
         assignment[i] = rand() % n_elems;
      }
      cuda_err_chk(cudaMalloc((void**)&d_assignment, settings.numReqs * 1024 * numblocks * sizeof(uint64_t)));
      cuda_err_chk(cudaMemcpy(d_assignment, assignment, settings.numReqs * 1024 * numblocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
      print_results("UVM_DIRECT_TEST RANDOM",
                    measure_performance([&]() {
                       for (int i = 0; i < (int)((data_size + DIRECTIO_MAX_RW_COUNT - 1) / DIRECTIO_MAX_RW_COUNT); i++) {
                          uint64_t itr_data_size = min(DIRECTIO_MAX_RW_COUNT, data_size - i * DIRECTIO_MAX_RW_COUNT);
                          uint64_t bytes2read = round_to_nearest(itr_data_size, 4096);
                          if (pread(fd, buf, bytes2read, i * DIRECTIO_MAX_RW_COUNT) != (ssize_t)bytes2read) {
                             perror("pread");
                             exit(EXIT_FAILURE);
                          }
                       }
                       naive_random_access_kernel<<<blockDim, n_threads>>>(
                           (uint64_t*)buf, d_assignment, numblocks * n_threads,
                           settings.numReqs);
                    }),
                    settings.numReqs * 1024 * numblocks * sizeof(uint64_t));
      free(assignment);
      cuda_err_chk(cudaFree(d_assignment));
   } else {
      print_results("UVM_DIRECT_TEST SEQUENTIAL",
                    measure_performance([&]() {
                       for (int i = 0; i < (int)((data_size + buf_size - 1) / buf_size); i++) {
                          uint64_t itr_data_size = min(buf_size, data_size - i * buf_size);
                          uint64_t bytes_read = round_to_nearest(itr_data_size, 4096);
                          if (pread(fd, buf, bytes_read, i * buf_size) != (ssize_t)bytes_read) {
                             perror("pread");
                             exit(EXIT_FAILURE);
                          }
                          naive_sequential_access_kernel<<<blockDim, n_threads>>>(
                              (uint64_t*)buf, numblocks * n_threads, itr_data_size / sizeof(uint64_t));
                          cudaDeviceSynchronize();
                       }
                    }),
                    data_size);
   }
   cudaFree(buf);
   close(fd);
   std::remove("./test.bin");
}

static void NV_GDS_TEST(Settings& settings) {
   if (settings.random) {
      fprintf(stderr, "NV_GDS_TEST does not support random access mode.\n");
      exit(EXIT_FAILURE);
   }

   size_t gpu_free_mem, total_mem;
   cuda_err_chk(cudaMemGetInfo(&gpu_free_mem, &total_mem));
   fprintf(stdout, "gpu_free_mem: %.2lf GiB, total_mem: %.2lf GiB \n", (double)gpu_free_mem / (1 << 30), (double)total_mem / (1 << 30));
   uint64_t n_elems = settings.numElems;
   uint64_t data_size = n_elems * sizeof(uint64_t);

   char* tmp_buf;
   uint64_t tmp_buf_size = round_to_nearest(min(data_size, (uint64_t)DIRECTIO_MAX_RW_COUNT), 4096);
   int fd = open("./test.bin", O_CREAT | O_RDWR | O_DIRECT, 0644);
   if (fd < 0) {
      perror("open");
      close(fd);
      exit(EXIT_FAILURE);
   }
   if (posix_memalign((void**)&tmp_buf, 4096, tmp_buf_size) != 0) {
      perror("posix_memalign");
      close(fd);
      std::remove("./test.bin");
      exit(EXIT_FAILURE);
   }
   if (getrandom(tmp_buf, tmp_buf_size, 0) == -1) {
      perror("getrandom");
      close(fd);
      std::remove("./test.bin");
      exit(EXIT_FAILURE);
   }
   for (int i = 0; i < (int)((data_size + tmp_buf_size - 1) / tmp_buf_size); i++) {
      uint64_t bytes2write = min(tmp_buf_size, data_size - i * tmp_buf_size);
      if (pwrite(fd, tmp_buf, bytes2write, i * tmp_buf_size) != (ssize_t)bytes2write) {
         perror("write");
         close(fd);
         std::remove("./test.bin");
         exit(EXIT_FAILURE);
      }
   }
   free(tmp_buf);

   cufile_err_chk(cuFileDriverOpen());
   uint64_t buf_size = round_to_nearest(min(data_size, (uint64_t)1 << 30), 4096), *buf;
   uint64_t n_threads = settings.numThreads;
   uint64_t numblocks = (buf_size / sizeof(uint64_t) + n_threads - 1) / n_threads;
   dim3 blockDim(numblocks);
   fprintf(stdout, "n_elems: %lu, data_size: %lu, buf_size: %lu\n",
           n_elems, data_size, buf_size);
   fprintf(stdout, "numblocks: %lu, n_threads: %lu\n", numblocks, n_threads);
   cuda_err_chk(cudaMalloc((void**)&buf, buf_size));
   CUfileDescr_t cf_descr;
   CUfileHandle_t cf_handle;
   memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
   cf_descr.handle.fd = fd;
   cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
   cufile_err_chk(cuFileHandleRegister(&cf_handle, &cf_descr));
   cufile_err_chk(cuFileBufRegister(buf, buf_size, 0));

   print_results("NV_GDS_TEST",
                 measure_performance([&]() {
                    for (int i = 0; i < (int)((data_size + buf_size - 1) / buf_size); i++) {
                       uint64_t itr_data_size = min(buf_size, data_size - i * buf_size);
                       cuFileRead(cf_handle, buf, itr_data_size, i * buf_size, 0);
                       naive_sequential_access_kernel<<<blockDim, n_threads>>>(
                           (uint64_t*)buf, numblocks * n_threads, itr_data_size / sizeof(uint64_t));
                    }
                 }),
                 data_size);

   cufile_err_chk(cuFileBufDeregister(buf));
   cuFileHandleDeregister(cf_handle);
   cufile_err_chk(cuFileDriverClose());
   cudaFree(buf);
   close(fd);
}

static void BAM_TEST(Settings& settings) {
   uint64_t b_size = settings.blkSize;                             // 64;
   uint64_t g_size = (settings.numThreads + b_size - 1) / b_size;  // 80*16;
   uint64_t n_threads = b_size * g_size;
   uint64_t page_size = settings.pageSize;
   uint64_t n_pages = settings.numPages;
   uint64_t total_cache_size = (page_size * n_pages);
   std::vector<Controller*> ctrls(settings.n_ctrls);
   for (size_t i = 0; i < settings.n_ctrls; i++) {
      ctrls[i] = new Controller(
          settings.ssdtype == 0 ? sam_ctrls_paths[i] : intel_ctrls_paths[i],
          settings.nvmNamespace, settings.cudaDevice, settings.queueDepth,
          settings.numQueues);
   }
   page_cache_t h_pc(page_size, n_pages, settings.cudaDevice, ctrls[0][0],
                     (uint64_t)64, ctrls);
   std::cout << "finished creating cache\n";
   // QueuePair* d_qp;
   page_cache_t* d_pc = (page_cache_t*)(h_pc.d_pc_ptr);
#define TYPE uint64_t
   uint64_t n_elems = settings.numElems;
   uint64_t t_size = n_elems * sizeof(TYPE);

   range_t<uint64_t> h_range((uint64_t)0, (uint64_t)n_elems, (uint64_t)0,
                             (uint64_t)(t_size / page_size), (uint64_t)0,
                             (uint64_t)page_size, &h_pc, settings.cudaDevice);
   range_t<uint64_t>* d_range = (range_t<uint64_t>*)h_range.d_range_ptr;

   std::vector<range_t<uint64_t>*> vr(1);
   vr[0] = &h_range;
   array_t<uint64_t> a(n_elems, 0, vr, settings.cudaDevice);
   std::cout << "finished creating range\n";

   unsigned long long* d_req_count;
   cuda_err_chk(cudaMalloc(&d_req_count, sizeof(unsigned long long)));
   cuda_err_chk(cudaMemset(d_req_count, 0, sizeof(unsigned long long)));
   std::cout << "atlaunch kernel\n";
   char st[15];
   cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, settings.cudaDevice));
   std::cout << st << std::endl;
   uint64_t *assignment, *d_assignment;
   if (settings.random) {
      assignment = (uint64_t*)malloc(n_threads * sizeof(uint64_t) * settings.numReqs);
      for (size_t i = 0; i < n_threads * settings.numReqs; i++) {
         assignment[i] = rand() % (n_elems);
      }
      cuda_err_chk(cudaMalloc(&d_assignment, settings.numReqs * n_threads * sizeof(uint64_t)));
      cuda_err_chk(cudaMemcpy(d_assignment, assignment,
                              n_threads * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));
   }
   Event before;
   if (settings.random) {
      bam_random_access_kernel<<<g_size, b_size>>>(
          a.d_array_ptr, n_threads, d_req_count, d_assignment, settings.numReqs, n_elems);
   } else {
      bam_sequential_access_kernel<<<g_size, b_size>>>(
          a.d_array_ptr, n_threads, d_req_count, n_elems);
   }
   Event after;
   cuda_err_chk(cudaDeviceSynchronize());
   double elapsed = after - before;
   uint64_t ios, data;
   if (settings.random) {
      ios = n_threads * settings.numReqs;
   } else {
      ios = n_elems;
   }
   data = ios * sizeof(uint64_t);
   double iops = ((double)ios) / (elapsed / 1000000);
   double bandwidth =
       (((double)data) / (elapsed / 1000000)) / (1024ULL * 1024ULL * 1024ULL);
   // uint64_t ios = g_size * b_size * settings.numReqs;
   // uint64_t data = ios * sizeof(uint64_t);
   // double iops = ((double)ios) / (elapsed / 1000000);
   // double bandwidth =
   //     (((double)data) / (elapsed / 1000000)) / (1024ULL * 1024ULL * 1024ULL);

   a.print_reset_stats();
   std::cout << "n_elems: " << n_elems << std::endl;
   std::cout << std::dec << "Elapsed Time: " << elapsed / 1e3
             << "ms\tNumber of Read Ops: " << ios
             << "\tData Size (bytes): " << data << std::endl;
   std::cout << std::dec << "Read Ops/sec: " << iops
             << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;

   print_results("BAM_TEST", elapsed / 1e3, data);
   if (settings.random) {
      free(assignment);
      cuda_err_chk(cudaFree(d_assignment));
   }
   for (size_t i = 0; i < settings.n_ctrls; i++) {
      delete ctrls[i];
   }
}

int main(int argc, char** argv) {
   Settings settings;
   try {
      settings.parseArguments(argc, argv);
   } catch (const string& e) {
      fprintf(stderr, "%s\n", e.c_str());
      fprintf(stderr, "%s\n", Settings::usageString(argv[0]).c_str());
      return 1;
   }

   cudaDeviceProp properties;
   if (cudaGetDeviceProperties(&properties, settings.cudaDevice) !=
       cudaSuccess) {
      fprintf(stderr, "Failed to get CUDA device properties\n");
      return 1;
   }

   mem_type mem = (mem_type)settings.memalloc;

   if (mem == GPUMEM) {
      GPUMEM_TEST(settings);
   } else if (mem == UVM_READONLY) {
      UVM_READONLY_TEST(settings);
   } else if (mem == UVM_DIRECT) {
      UVM_DIRECT_TEST(settings);
   } else if (mem == NV_GDS) {
      NV_GDS_TEST(settings);
   } else if (mem == BAFS_DIRECT) {
      BAM_TEST(settings);
   } else {
      fprintf(stderr, "Using memory type: %d\n", mem);
      exit(EXIT_FAILURE);
   }

   return 0;
}
