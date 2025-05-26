cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

make_benchmark (vectoradd-benchmark-module vectoradd-bench "main.cu;")

target_link_libraries(vectoradd-benchmark-module
    cufile
    ${CUDA_CUDART_LIBRARY}
    Threads::Threads
)

make_benchmark_choice (vectoradd-benchmark vectoradd-benchmark-smartio vectoradd-benchmark-module)
