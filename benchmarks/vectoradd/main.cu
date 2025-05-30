/* References:
 *
 *      Coalesce
 *          Hong, Sungpack, et al.
 *          "Accelerating CUDA graph algorithms at maximum warp."
 *          Acm Sigplan Notices 46.8 (2011): 267-276.
 *
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <iostream>
// #include "helper_cuda.h"
#include <buffer.h>
#include <ctrl.h>
#include <event.h>
#include <fcntl.h>
#include <math.h>
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
#include <unistd.h>
#include <util.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <iterator>
#include <numeric>
#include <ratio>
#include <stdexcept>
#include <vector>

#include "settings.h"

#define UINT64MAX 0xFFFFFFFFFFFFFFFF
#define DIRECTIO_MAX_RW_COUNT ((1ULL << 31) - 4096)

using error = std::runtime_error;
using std::string;
// const char* const ctrls_paths[] = {"/dev/libnvmpro0", "/dev/libnvmpro1",
// "/dev/libnvmpro2", "/dev/libnvmpro3", "/dev/libnvmpro4", "/dev/libnvmpro5",
// "/dev/libnvmpro6", "/dev/libnvmpro7"};
const char *const ctrls_paths[] = {
    "/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3",
    "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7",
    "/dev/libnvm8", "/dev/libnvm9"};

#define WARP_SHIFT 5
#define WARP_SIZE 32

#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)

#define BLOCK_NUM 1024ULL

#define MAXWARP 64

typedef uint64_t EdgeT;

typedef enum {
   BASELINE = 0,
   OPTIMIZED = 1,
   BASELINE_PC = 2,
   OPTIMIZED_PC = 3,
} impl_type;

typedef enum {
   GPUMEM = 0,
   UVM_READONLY = 1,
   // UVM_DIRECT = 2,
   NV_GDS = 3,
   // UVM_READONLY_NVLINK = 3,
   // UVM_DIRECT_NVLINK = 4,
   BAFS_DIRECT = 6,
} mem_type;

__global__  //__launch_bounds__(64,32)
    void
    kernel_baseline(uint64_t n_elems, uint64_t *A, uint64_t *B, uint64_t *sum) {
   // uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x *
   // blockIdx.x + threadIdx.x;
   uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
   if (tid < n_elems) {
      sum[tid] = A[tid] + B[tid];
      // uint64_t val = A[tid] + B[tid];
      // atomicAdd(&sum[0], val);
      // printf("tid: %llu A:%llu B:%llu \n",tid,  A[tid], B[tid]);
   }
}

__global__ __launch_bounds__(64, 32) void kernel_baseline_ptr_pc(
    array_d_t<uint64_t> *da, array_d_t<uint64_t> *db, uint64_t n_elems,
    array_d_t<uint64_t> *dc, unsigned long long int *sum) {
   uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

   bam_ptr<uint64_t> Aptr(da);
   bam_ptr<uint64_t> Bptr(db);
   bam_ptr<uint64_t> Cptr(dc);

   if (tid < n_elems) {
      Cptr[tid] = Aptr[tid] + Bptr[tid];
      // uint64_t val = Aptr[tid] + Bptr[tid];
      // uint64_t val = A[tid] + B[tid];
      // sum[tid] = val;
      // atomicAdd(&sum[0], val);
   }
}

template <typename T>
__global__ __launch_bounds__(64, 32) void kernel_sequential_warp(
    T *A, T *B, uint64_t n_elems, uint64_t n_pages_per_warp, T *sum,
    uint64_t n_warps, size_t page_size) {
   const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
   const uint64_t lane = tid % 32;
   const uint64_t warp_id = tid / 32;
   const uint64_t n_elems_per_page = page_size / sizeof(T);
   T val = 0;
   uint64_t idx = 0;

   if (tid == 0) printf("n_elems_per_page: %llu\n", n_elems_per_page);
   if (warp_id < n_warps) {
      size_t start_page = n_pages_per_warp * warp_id;
      ;
      for (size_t i = 0; i < n_pages_per_warp; i++) {
         size_t cur_page = start_page + i;
         size_t start_idx = cur_page * n_elems_per_page + lane;

         for (size_t j = 0; j < n_elems_per_page; j += WARP_SIZE) {
            idx = start_idx + j;
            if (idx < n_elems) {
               val = A[idx] + B[idx];
               sum[idx] = val;
               // atomicAdd(&sum[0], val);
               //            printf("tid: %llu A:%llu B:%llu \n",idx,  A[tid],
               //            B[tid]);
            }
         }
      }
   }
}

template <typename T>
__global__  //__launch_bounds__(64,32)
    void
    kernel_sequential_warp_ptr_pc(array_d_t<T> *da, array_d_t<T> *db,
                                  uint64_t n_elems, uint64_t n_pages_per_warp,
                                  array_d_t<T> *dc, unsigned long long *sum,
                                  uint64_t n_warps, size_t page_size,
                                  uint64_t stride) {
   bam_ptr<uint64_t> Aptr(da);
   bam_ptr<uint64_t> Bptr(db);
   bam_ptr<uint64_t> Cptr(dc);

   const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
   const uint64_t lane = tid % 32;
   const uint64_t old_warp_id = tid / 32;
   const uint64_t n_elems_per_page = page_size / sizeof(T);
   T val = 0;
   uint64_t idx = 0;
   uint64_t nep = (n_warps + stride - 1) / stride;
   uint64_t warp_id = (old_warp_id / nep) + ((old_warp_id % nep) * stride);

   if (warp_id < n_warps) {
      size_t start_page = n_pages_per_warp * warp_id;
      ;
      for (size_t i = 0; i < n_pages_per_warp; i++) {
         size_t cur_page = start_page + i;
         size_t start_idx = cur_page * n_elems_per_page + lane;

         for (size_t j = 0; j < n_elems_per_page; j += WARP_SIZE) {
            idx = start_idx + j;
            if (idx < n_elems) {
               val = Aptr[idx] + Bptr[idx];
               Cptr[idx] = val;
               // sum[idx] = val;
               // atomicAdd(&sum[0], val);
               //            printf("tid: %llu A:%llu B:%llu \n",idx,  A[tid],
               //            B[tid]);
            }
         }
      }
   }
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

static uint64_t round_to_nearest(uint64_t x, uint64_t target) {
   return ((x + target - 1) / target) * target;
}

static void GPUMEM_TEST(const Settings &settings, size_t gpu_free_mem) {
   impl_type type = (impl_type)settings.type;
   uint64_t numthreads = settings.numThreads, n_elems = settings.n_elems;
   uint64_t n_elems_size = n_elems * sizeof(uint64_t), buf_size;
   uint64_t *a_h, *b_h, *a_d, *b_d, *sum_d, numblocks, n_warps;

   gpu_free_mem = (size_t)((double)gpu_free_mem * 0.9);
   buf_size = min((unsigned long)DIRECTIO_MAX_RW_COUNT, min(n_elems_size, gpu_free_mem / 3));
   buf_size = round_to_nearest(buf_size, 4096);
   fprintf(stdout, "gpu_free_mem: %lu, buf_size: %lu\n", gpu_free_mem, buf_size);

   if (type == BASELINE) {
      numblocks = (((buf_size / sizeof(uint64_t)) + numthreads - 1) / numthreads);
      fprintf(stdout, "numblocks: %lu, numthreads: %lu\n", numblocks, numthreads);
   } else if (type == OPTIMIZED) {
      uint64_t n_elems_per_page = settings.pageSize / sizeof(uint64_t);
      n_warps = ((buf_size / sizeof(uint64_t)) + n_elems_per_page - 1) /
                n_elems_per_page;
      numblocks = (n_warps * WARP_SIZE + numthreads - 1) / numthreads;
      fprintf(stdout, "page_size: %lu, n_elems_per_page: %lu\n",
              settings.pageSize, settings.pageSize / sizeof(uint64_t));
      fprintf(stdout, "numblocks: %lu, n_warps: %lu, numthreads: %lu\n",
              numblocks, n_warps, numthreads);
   } else {
      fprintf(stderr, "Unsupported impl_type\n");
      exit(EXIT_FAILURE);
   }
   dim3 blockDim(numblocks);

   if (posix_memalign((void **)&a_h, 4096, buf_size) != 0 ||
       posix_memalign((void **)&b_h, 4096, buf_size) != 0) {
      std::cerr << "posix_memalign failed\n";
      exit(EXIT_FAILURE);
   }
   cuda_err_chk(cudaHostRegister(a_h, buf_size, cudaHostRegisterDefault));
   cuda_err_chk(cudaHostRegister(b_h, buf_size, cudaHostRegisterDefault));
   cuda_err_chk(cudaMalloc((void **)&a_d, buf_size));
   cuda_err_chk(cudaMalloc((void **)&b_d, buf_size));
   cuda_err_chk(cudaMalloc((void **)&sum_d, buf_size));

   auto start = std::chrono::high_resolution_clock::now();
   int fda = open(settings.input_a, O_RDONLY | O_DIRECT);
   if (fda < 0) {
      fprintf(stderr, "Error: Failed to open file %s\n", settings.input_a);
      free(a_h);
      free(b_h);
      cudaFree(a_d);
      cudaFree(b_d);
      exit(EXIT_FAILURE);
   }
   int fdb = open(settings.input_b, O_RDONLY | O_DIRECT);
   if (fdb < 0) {
      fprintf(stderr, "Error: Failed to open file %s\n", settings.input_b);
      free(a_h);
      free(b_h);
      cudaFree(a_d);
      cudaFree(b_d);
      close(fda);
      exit(EXIT_FAILURE);
   }
   int fdc = open("./tmp_output.bin", O_WRONLY | O_CREAT | O_DIRECT, 0644);
   if (fdc < 0) {
      fprintf(stderr, "Error: Failed to open file ../tmp_output.bin\n");
      free(a_h);
      free(b_h);
      cudaFree(a_d);
      cudaFree(b_d);
      close(fda);
      close(fdb);
      exit(EXIT_FAILURE);
   }

   for (uint64_t i = 0; i < n_elems_size + 16; i += buf_size) {
      uint64_t itr_n_elems_size = min(buf_size, n_elems_size + 16 - i);
      uint64_t bytes2read = round_to_nearest(itr_n_elems_size, 4096);
      ssize_t ret;
      if ((ret = read(fda, a_h, bytes2read)) != bytes2read) {
         fprintf(stderr, "ret: %lld, bytes2read: %lld\n", ret, bytes2read);
         perror("pread");
      }
      if ((ret = read(fdb, b_h, bytes2read)) != bytes2read) {
         fprintf(stderr, "ret: %lld, bytes2read: %lld\n", ret, bytes2read);
         perror("pread");
      }
      if (i == 0) {
         itr_n_elems_size -= 16;
         cudaMemcpy(a_d, a_h + 2, itr_n_elems_size, cudaMemcpyHostToDevice);
         cudaMemcpy(b_d, b_h + 2, itr_n_elems_size, cudaMemcpyHostToDevice);
      } else {
         cudaMemcpy(a_d, a_h, itr_n_elems_size, cudaMemcpyHostToDevice);
         cudaMemcpy(b_d, b_h, itr_n_elems_size, cudaMemcpyHostToDevice);
      }
      if (type == BASELINE) {
         kernel_baseline<<<blockDim, numthreads>>>(
             itr_n_elems_size / sizeof(uint64_t), a_d, b_d, sum_d);
      } else if (type == OPTIMIZED) {
         kernel_sequential_warp<uint64_t><<<blockDim, numthreads>>>(
             a_d, b_d, itr_n_elems_size / sizeof(uint64_t), 1, sum_d, n_warps,
             settings.pageSize);
      }
      cuda_err_chk(
          cudaMemcpy(a_h, sum_d, itr_n_elems_size, cudaMemcpyDeviceToHost));
      if ((ret = write(fdc, a_h, bytes2read)) != bytes2read) {
         fprintf(stderr, "ret: %lld, bytes2read: %lld\n", ret, bytes2read);
         perror("pwrite");
      }
   }
   cudaDeviceSynchronize();

   auto end = std::chrono::high_resolution_clock::now();
   fprintf(stdout, "Elapsed Time: %ld ms\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
               .count());

   free(a_h);
   free(b_h);
   cuda_err_chk(cudaFree(a_d));
   cuda_err_chk(cudaFree(b_d));
   cuda_err_chk(cudaFree(sum_d));
   close(fdc);
   close(fda);
   close(fdb);
   std::remove("./tmp_output.bin");
}

static void UVM_READONLY_TEST(const Settings &settings, uint64_t sys_free_mem) {
   std::ifstream filea, fileb;
   std::ofstream filec;
   impl_type type = (impl_type)settings.type;
   uint64_t numthreads = settings.numThreads, n_elems = settings.n_elems;
   uint64_t n_elems_size = n_elems * sizeof(uint64_t), buf_size, tmp[2],
            numblocks, n_warps;
   uint64_t *a, *b, *sum;

   sys_free_mem = (size_t)((double)sys_free_mem * 0.5);
   if (n_elems_size < sys_free_mem / 3) {
      buf_size = n_elems_size;
   } else {
      buf_size = sys_free_mem / 3;
   }
   buf_size = round_to_nearest(buf_size, 4096);

   fprintf(stdout, "sys_free_mem: %lu, buf_size: %lu\n", sys_free_mem, buf_size);
   if (type == BASELINE) {
      numblocks = (((buf_size / sizeof(uint64_t)) + numthreads - 1) / numthreads);
      fprintf(stdout, "numblocks: %lu, numthreads: %lu\n", numblocks, numthreads);
   } else if (type == OPTIMIZED) {
      uint64_t n_elems_per_page = settings.pageSize / sizeof(uint64_t);
      n_warps = ((buf_size / sizeof(uint64_t)) + n_elems_per_page - 1) /
                n_elems_per_page;
      numblocks = (n_warps * WARP_SIZE + numthreads - 1) / numthreads;
      fprintf(stdout, "page_size: %lu, n_elems_per_page: %lu\n",
              settings.pageSize, settings.pageSize / sizeof(uint64_t));
      fprintf(stdout, "numblocks: %lu, n_warps: %lu, numthreads: %lu\n",
              numblocks, n_warps, numthreads);
   } else {
      fprintf(stderr, "Unsupported impl_type\n");
      exit(EXIT_FAILURE);
   }

   dim3 blockDim(numblocks);
   cuda_err_chk(cudaMallocManaged((void **)&a, buf_size));
   cuda_err_chk(cudaMallocManaged((void **)&b, buf_size));
   cuda_err_chk(cudaMallocManaged((void **)&sum, buf_size));
   cuda_err_chk(cudaMemAdvise(a, buf_size, cudaMemAdviseSetAccessedBy,
                              settings.cudaDevice));
   cuda_err_chk(cudaMemAdvise(b, buf_size, cudaMemAdviseSetAccessedBy,
                              settings.cudaDevice));

   auto start = std::chrono::high_resolution_clock::now();
   filea.open(settings.input_a, std::ios::in | std::ios::binary);
   if (!filea.is_open()) {
      fprintf(stderr, "Error: Failed to open file %s\n", settings.input_a);
      exit(EXIT_FAILURE);
   }
   fileb.open(settings.input_b, std::ios::in | std::ios::binary);
   if (!fileb.is_open()) {
      fprintf(stderr, "Error: Failed to open file %s\n", settings.input_b);
      filea.close();
      exit(EXIT_FAILURE);
   }
   filec.open("./tmp_output.bin", std::ios::out | std::ios::binary);
   if (!filec.is_open()) {
      fprintf(stderr, "Error: Failed to open file ./tmp_output.bin\n");
      filea.close();
      exit(EXIT_FAILURE);
   }
   filea.read((char *)tmp, 16);
   fileb.read((char *)tmp, 16);

   for (uint64_t i = 0; i < n_elems_size; i += buf_size) {
      uint64_t itr_n_elems_size = min(buf_size, n_elems_size - i);
      filea.read((char *)a, itr_n_elems_size);
      fileb.read((char *)b, itr_n_elems_size);
      if (type == BASELINE) {
         kernel_baseline<<<blockDim, numthreads>>>(
             itr_n_elems_size / sizeof(uint64_t), a, b, sum);
      } else if (type == OPTIMIZED) {
         kernel_sequential_warp<uint64_t>
             <<<blockDim, numthreads>>>(a, b, itr_n_elems_size / sizeof(uint64_t),
                                        1, sum, n_warps, settings.pageSize);
      }
      filec.write((char *)sum, itr_n_elems_size);
   }
   cudaDeviceSynchronize();

   auto end = std::chrono::high_resolution_clock::now();
   fprintf(stdout, "Elapsed Time: %ld ms\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
               .count());

   cudaFree(a);
   cudaFree(b);
   cudaFree(sum);
   filea.close();
   fileb.close();
   filec.close();
   std::remove("./tmp_output.bin");
}

static void UVM_DIRECT_TEST(const Settings &settings, uint64_t sys_free_mem) {
   impl_type type = (impl_type)settings.type;
   uint64_t numthreads = settings.numThreads, n_elems = settings.n_elems;
   uint64_t n_elems_size = n_elems * sizeof(uint64_t), buf_size, tmp[2],
            numblocks, n_warps;
   uint64_t *a, *b, *sum;

   sys_free_mem = (size_t)((double)sys_free_mem * 0.5);
   if (n_elems_size < sys_free_mem / 3) {
      buf_size = n_elems_size;
   } else {
      buf_size = sys_free_mem / 3;
   }
   buf_size = round_to_nearest(buf_size, 4096);

   fprintf(stdout, "sys_free_mem: %lu, buf_size: %lu\n", sys_free_mem, buf_size);
   if (type == BASELINE) {
      numblocks = (((buf_size / sizeof(uint64_t)) + numthreads - 1) / numthreads);
      fprintf(stdout, "numblocks: %lu, numthreads: %lu\n", numblocks, numthreads);
   } else if (type == OPTIMIZED) {
      uint64_t n_elems_per_page = settings.pageSize / sizeof(uint64_t);
      n_warps = ((buf_size / sizeof(uint64_t)) + n_elems_per_page - 1) /
                n_elems_per_page;
      numblocks = (n_warps * WARP_SIZE + numthreads - 1) / numthreads;
      fprintf(stdout, "page_size: %lu, n_elems_per_page: %lu\n",
              settings.pageSize, settings.pageSize / sizeof(uint64_t));
      fprintf(stdout, "numblocks: %lu, n_warps: %lu, numthreads: %lu\n",
              numblocks, n_warps, numthreads);
   } else {
      fprintf(stderr, "Unsupported impl_type\n");
      exit(EXIT_FAILURE);
   }

   dim3 blockDim(numblocks);
   cuda_err_chk(cudaMallocManaged((void **)&a, buf_size));
   cuda_err_chk(cudaMallocManaged((void **)&b, buf_size));
   cuda_err_chk(cudaMallocManaged((void **)&sum, buf_size));
   cuda_err_chk(cudaMemAdvise(a, buf_size, cudaMemAdviseSetReadMostly,
                              settings.cudaDevice));
   cuda_err_chk(cudaMemAdvise(b, buf_size, cudaMemAdviseSetReadMostly,
                              settings.cudaDevice));

   auto start = std::chrono::high_resolution_clock::now();
   int fda = open(settings.input_a, O_RDONLY | O_DIRECT);
   if (fda == -1) {
      fprintf(stderr, "Error: Failed to open file %s\n", settings.input_a);
      exit(EXIT_FAILURE);
   }
   int fdb = open(settings.input_b, O_RDONLY | O_DIRECT);
   if (fdb == -1) {
      fprintf(stderr, "Error: Failed to open file %s\n", settings.input_b);
      exit(EXIT_FAILURE);
   }
   int fdc = open("./tmp_output.bin", O_WRONLY | O_CREAT | O_DIRECT, 0644);
   if (fdc == -1) {
      fprintf(stderr, "Error: Failed to open file ./tmp_output.bin\n");
      close(fda);
      close(fdb);
      exit(EXIT_FAILURE);
   }

   for (uint64_t i = 0; i < (n_elems_size + 16 + buf_size - 1) / buf_size; i++) {
      uint64_t itr_n_elems_size = min(buf_size, n_elems_size + 16 - i * buf_size);
      uint64_t bytes2read = round_to_nearest(itr_n_elems_size, 4096);
      ssize_t ret;
      fprintf(stdout, "%lld %lld %lld\n", itr_n_elems_size, bytes2read, buf_size);
      if ((ret = pread(fda, a, bytes2read, i * buf_size)) != bytes2read) {
         perror("pread");
      }
      if ((ret = pread(fdb, b, bytes2read, i * buf_size)) != bytes2read) {
         perror("pread");
      }
      if (i == 0) {
         itr_n_elems_size -= 16;
         if (type == BASELINE) {
            kernel_baseline<<<blockDim, numthreads>>>(
                itr_n_elems_size / sizeof(uint64_t), a + 2, b + 2, sum);
         } else if (type == OPTIMIZED) {
            kernel_sequential_warp<uint64_t>
                <<<blockDim, numthreads>>>(a + 2, b + 2, itr_n_elems_size / sizeof(uint64_t),
                                           1, sum, n_warps, settings.pageSize);
         }
      } else {
         if (type == BASELINE) {
            kernel_baseline<<<blockDim, numthreads>>>(
                itr_n_elems_size / sizeof(uint64_t), a, b, sum);
         } else if (type == OPTIMIZED) {
            kernel_sequential_warp<uint64_t>
                <<<blockDim, numthreads>>>(a, b, itr_n_elems_size / sizeof(uint64_t),
                                           1, sum, n_warps, settings.pageSize);
         }
      }
      if ((ret = pwrite(fdc, sum, bytes2read, i * buf_size)) = bytes2read) {
         perror("pwrite");
      }
   }
   cudaDeviceSynchronize();

   auto end = std::chrono::high_resolution_clock::now();
   fprintf(stdout, "Elapsed Time: %ld ms\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
               .count());

   cudaFree(a);
   cudaFree(b);
   cudaFree(sum);
   close(fda);
   close(fdb);
   close(fdc);
   std::remove("./tmp_output.bin");
}

static void NV_GDS_TEST(const Settings &settings, size_t gpu_free_mem) {
   impl_type type = (impl_type)settings.type;
   uint64_t numthreads = settings.numThreads, n_elems = settings.n_elems;
   uint64_t n_elems_size = n_elems * sizeof(uint64_t), buf_size;
   uint64_t *a, *b, *sum, numblocks, n_warps;

   cufile_err_chk(cuFileDriverOpen());

   gpu_free_mem = (size_t)((double)gpu_free_mem * 0.9);
   if (n_elems_size < gpu_free_mem / 3) {
      buf_size = n_elems_size;
   } else {
      buf_size = gpu_free_mem / 3;
   }
   buf_size = round_to_nearest(buf_size, 4096);
   fprintf(stdout, "gpu_free_mem: %lu, buf_size: %lu\n", gpu_free_mem, buf_size);
   if (type == BASELINE) {
      numblocks = (((buf_size / sizeof(uint64_t)) + numthreads - 1) / numthreads);
      fprintf(stdout, "numblocks: %lu, numthreads: %lu\n", numblocks, numthreads);
   } else if (type == OPTIMIZED) {
      uint64_t n_elems_per_page = settings.pageSize / sizeof(uint64_t);
      n_warps = ((buf_size / sizeof(uint64_t)) + n_elems_per_page - 1) /
                n_elems_per_page;
      numblocks = (n_warps * WARP_SIZE + numthreads - 1) / numthreads;
      fprintf(stdout, "page_size: %lu, n_elems_per_page: %lu\n",
              settings.pageSize, settings.pageSize / sizeof(uint64_t));
      fprintf(stdout, "numblocks: %lu, n_warps: %lu, numthreads: %lu\n",
              numblocks, n_warps, numthreads);
   } else {
      fprintf(stderr, "Unsupported impl_type\n");
      exit(EXIT_FAILURE);
   }
   dim3 blockDim(numblocks);
   cuda_err_chk(cudaMalloc((void **)&a, buf_size));
   cuda_err_chk(cudaMalloc((void **)&b, buf_size));
   cuda_err_chk(cudaMalloc((void **)&sum, buf_size));

   auto start = std::chrono::high_resolution_clock::now();
   int fda = open(settings.input_a, O_RDONLY | O_DIRECT);
   if (fda == -1) {
      fprintf(stderr, "Error: Failed to open file %s\n", settings.input_a);
      exit(EXIT_FAILURE);
   }
   int fdb = open(settings.input_b, O_RDONLY | O_DIRECT);
   if (fdb == -1) {
      fprintf(stderr, "Error: Failed to open file %s\n", settings.input_b);
      close(fda);
      exit(EXIT_FAILURE);
   }
   int fdc = open("./tmp_output.bin", O_WRONLY | O_CREAT | O_DIRECT, 0644);
   if (fdc == -1) {
      fprintf(stderr, "Error: Failed to open file ./tmp_output.bin\n");
      close(fda);
      close(fdb);
      exit(EXIT_FAILURE);
   }
   CUfileDescr_t cf_descr_a, cf_descr_b, cf_descr_c;
   CUfileHandle_t cf_handle_a, cf_handle_b, cf_handle_c;
   memset((void *)&cf_descr_a, 0, sizeof(CUfileDescr_t));
   memset((void *)&cf_descr_b, 0, sizeof(CUfileDescr_t));
   memset((void *)&cf_descr_c, 0, sizeof(CUfileDescr_t));
   cf_descr_a.handle.fd = fda;
   cf_descr_b.handle.fd = fdb;
   cf_descr_c.handle.fd = fdc;
   cf_descr_a.type = cf_descr_b.type = cf_descr_c.type =
       CU_FILE_HANDLE_TYPE_OPAQUE_FD;
   cufile_err_chk(cuFileHandleRegister(&cf_handle_a, &cf_descr_a));
   cufile_err_chk(cuFileHandleRegister(&cf_handle_b, &cf_descr_b));
   cufile_err_chk(cuFileHandleRegister(&cf_handle_c, &cf_descr_c));
   cufile_err_chk(cuFileBufRegister(a, buf_size, 0));
   cufile_err_chk(cuFileBufRegister(b, buf_size, 0));
   cufile_err_chk(cuFileBufRegister(sum, buf_size, 0));

   for (uint64_t i = 0; i < n_elems_size; i += buf_size) {
      uint64_t itr_n_elems_size = min(buf_size, n_elems_size - i);
      cuFileRead(cf_handle_a, a, itr_n_elems_size, 16 + i, 0);
      cuFileRead(cf_handle_b, b, itr_n_elems_size, 16 + i, 0);
      if (type == BASELINE) {
         kernel_baseline<<<blockDim, numthreads>>>(
             itr_n_elems_size / sizeof(uint64_t), a, b, sum);
      } else if (type == OPTIMIZED) {
         kernel_sequential_warp<uint64_t>
             <<<blockDim, numthreads>>>(a, b, itr_n_elems_size / sizeof(uint64_t),
                                        1, sum, n_warps, settings.pageSize);
      }
      cuFileWrite(cf_handle_c, sum, itr_n_elems_size, i, 0);
   }
   cudaDeviceSynchronize();

   auto end = std::chrono::high_resolution_clock::now();
   fprintf(stdout, "Elapsed Time: %ld ms\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
               .count());

   cuFileDriverClose();
   cudaFree(a);
   cudaFree(b);
   cudaFree(sum);
   close(fda);
   close(fdb);
   close(fdc);
   cuFileHandleDeregister(cf_handle_a);
   cuFileHandleDeregister(cf_handle_b);
   cuFileHandleDeregister(cf_handle_c);
   std::remove("./tmp_output.bin");
}

static void BAM_TEST(const Settings &settings, size_t gpu_free_mem) {
   impl_type type = (impl_type)settings.type;
   uint64_t numthreads = settings.numThreads, n_elems = settings.n_elems;
   uint64_t n_elems_size = n_elems * sizeof(uint64_t);
   uint64_t pc_page_size = settings.pageSize,
            pc_pages = ceil((float)settings.maxPageCacheSize / pc_page_size);
   uint64_t n_pages = ceil(((float)n_elems_size) / pc_page_size), numblocks,
            n_warps;

   fprintf(stdout, "page size: %lu, pc_entries: %lu, n_pages: %lu\n",
           pc_page_size, pc_pages, n_pages);
   std::vector<Controller *> ctrls(settings.n_ctrls);
   for (size_t i = 0; i < settings.n_ctrls; i++) {
      ctrls[i] = new Controller(ctrls_paths[i], settings.nvmNamespace,
                                settings.cudaDevice, settings.queueDepth,
                                settings.numQueues);
   }
   fprintf(stdout, "Controllers Created\n");
   if (type == BASELINE_PC) {
      numblocks = ((n_elems + numthreads - 1) / numthreads);
   } else if (type == OPTIMIZED_PC) {
      uint64_t n_elems_per_page = pc_page_size / sizeof(uint64_t);
      n_warps = (n_elems + n_elems_per_page - 1) / n_elems_per_page;
      numblocks = (n_warps * WARP_SIZE + numthreads - 1) / numthreads;
   } else {
      fprintf(stderr, "Unsupported impl_type\n");
      exit(EXIT_FAILURE);
   }
   fprintf(stdout, "numblocks: %llu, numthreads: %llu\n", numblocks, numthreads);
   dim3 blockDim(numblocks);
   uint64_t cfileoffset = 720 * 1024 * 1024 * 1024;
   page_cache_t *h_pc =
       new page_cache_t(pc_page_size, pc_pages, settings.cudaDevice, ctrls[0][0],
                        (uint64_t)64, ctrls);
   range_t<uint64_t> *h_Arange = new range_t<uint64_t>(
       (uint64_t)0, (uint64_t)n_elems,
       (uint64_t)(ceil(settings.afileoffset * 1.0 / pc_page_size)),
       (uint64_t)n_pages, (uint64_t)0, (uint64_t)pc_page_size, h_pc,
       settings.cudaDevice);
   range_t<uint64_t> *h_Brange = new range_t<uint64_t>(
       (uint64_t)0, (uint64_t)n_elems,
       (uint64_t)(ceil(settings.bfileoffset * 1.0 / pc_page_size)),
       (uint64_t)n_pages, (uint64_t)0, (uint64_t)pc_page_size, h_pc,
       settings.cudaDevice);
   range_t<uint64_t> *h_Crange = new range_t<uint64_t>(
       (uint64_t)0, (uint64_t)n_elems,
       (uint64_t)(ceil(cfileoffset * 1.0 / pc_page_size)), (uint64_t)n_pages,
       (uint64_t)0, (uint64_t)pc_page_size, h_pc, settings.cudaDevice);
   std::vector<range_t<uint64_t> *> vec_Arange({h_Arange});
   std::vector<range_t<uint64_t> *> vec_Brange({h_Brange});
   std::vector<range_t<uint64_t> *> vec_Crange({h_Crange});
   array_t<uint64_t> *h_Aarray = new array_t<uint64_t>(
       n_elems, settings.afileoffset, vec_Arange, settings.cudaDevice);
   array_t<uint64_t> *h_Barray = new array_t<uint64_t>(
       n_elems, settings.bfileoffset, vec_Brange, settings.cudaDevice);
   array_t<uint64_t> *h_Carray = new array_t<uint64_t>(
       n_elems, cfileoffset, vec_Crange, settings.cudaDevice);
   fprintf(stdout, "Page cache initialized\n");

   auto start = std::chrono::high_resolution_clock::now();
   if (type == BASELINE_PC) {
      printf(
          "launching baseline_pc: blockDim.x :%llu blockDim.y :%llu "
          "numthreads:%llu\n",
          blockDim.x, blockDim.y, numthreads);
      kernel_baseline_ptr_pc<<<blockDim, numthreads>>>(
          h_Aarray->d_array_ptr, h_Barray->d_array_ptr, n_elems,
          h_Carray->d_array_ptr, nullptr);
      h_pc->flush_cache();
   } else if (type == OPTIMIZED_PC) {
      printf("launching optimized: blockDim.x :%llu numthreads:%llu\n",
             blockDim.x, numthreads);
      kernel_sequential_warp_ptr_pc<uint64_t>
          <<<blockDim, numthreads>>>(h_Aarray->d_array_ptr, h_Barray->d_array_ptr,
                                     n_elems, 1, h_Carray->d_array_ptr, nullptr,
                                     n_warps, settings.pageSize, settings.stride);
      h_pc->flush_cache();
   }
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }
   cuda_err_chk(cudaDeviceSynchronize());
   auto end = std::chrono::high_resolution_clock::now();
   h_Aarray->print_reset_stats();
   h_Barray->print_reset_stats();
   fprintf(stdout, "Elapsed Time: %ld ms\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
               .count());

   delete h_pc;
   delete h_Arange;
   delete h_Brange;
   delete h_Crange;
   delete h_Aarray;
   delete h_Barray;
   delete h_Carray;
   for (size_t i = 0; i < settings.n_ctrls; i++) {
      delete ctrls[i];
   }
}

int main(int argc, char *argv[]) {
   Settings settings;
   try {
      settings.parseArguments(argc, argv);
   } catch (const string &e) {
      fprintf(stderr, "%s\n", e.c_str());
      fprintf(stderr, "%s\n", Settings::usageString(argv[0]).c_str());
      exit(EXIT_FAILURE);
   }

   cudaDeviceProp properties;
   if (cudaGetDeviceProperties(&properties, settings.cudaDevice) !=
       cudaSuccess) {
      fprintf(stderr, "Failed to get CUDA device properties\n");
      exit(EXIT_FAILURE);
   }

   mem_type mem = (mem_type)settings.memalloc;
   impl_type type = (impl_type)settings.type;
   size_t cpu_free_mem, gpu_free_mem, total_mem;
   cpu_free_mem = get_cpu_free_mem();
   cuda_err_chk(cudaMemGetInfo(&gpu_free_mem, &total_mem));

   fprintf(stdout, "********* System Info ****************\n");
   fprintf(stdout, "GPU Name: %s\n", properties.name);
   fprintf(stdout, "cpu_free_mem: %.2lf GiB\n",
           (double)cpu_free_mem / (1 << 30));
   fprintf(stdout, "gpu_free_mem: %.2lf GiB, total_mem: %.2lf GiB \n",
           (double)gpu_free_mem / (1 << 30), (double)total_mem / (1 << 30));
   fprintf(stdout, "**************************************\n");

   fprintf(stdout, "********* Test Info ******************\n");
   fprintf(stdout, "file A: %s, file B: %s\n", settings.input_a,
           settings.input_b);
   fprintf(stdout, "mem_type: %d, impl_type: %d\n", mem, type);
   fprintf(stdout, "n_elements: %lu, n_elems_size: %lu\n", settings.n_elems,
           settings.n_elems * sizeof(uint64_t));
   fprintf(stdout, "**************************************\n");

   if (mem == GPUMEM) {
      GPUMEM_TEST(settings, gpu_free_mem);
   } else if (mem == UVM_READONLY) {
      UVM_READONLY_TEST(settings, cpu_free_mem + gpu_free_mem);
   }
   //  else if (mem == UVM_DIRECT) {
   //     UVM_DIRECT_TEST(settings, cpu_free_mem + gpu_free_mem);
   //  }
   else if (mem == NV_GDS) {
      NV_GDS_TEST(settings, gpu_free_mem);
   } else if (mem == BAFS_DIRECT) {
      BAM_TEST(settings, gpu_free_mem);
   } else {
      fprintf(stderr, "Unsupport mem_type\n");
      exit(EXIT_FAILURE);
   }

   return 0;
}
