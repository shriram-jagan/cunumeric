/* Copyright 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "cunumeric/matrix/solve_tridiagonal.h"
#include "cunumeric/matrix/solve_tridiagonal_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace legate;

template <typename Getgtsv2BufferSize, typename Getgtsv2Solver, typename VAL>
static inline void solve_batched_tridiagonal_template(Getgtsv2BufferSize gtsv2_buffer_size, 
                                  Getgtsv2Solver gtsv2solver,
                                  int32_t linear_system_size,
                                  VAL *dl,
                                  VAL *d,
                                  VAL *du, 
                                  VAL *x,
                                  int32_t batch_count,
                                  int32_t batch_stride)
{
  auto handle = get_cusparse(); 
  auto stream = get_cached_stream();
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));

  size_t buffer_size;

  CHECK_CUSPARSE(gtsv2_buffer_size(handle, linear_system_size, dl, d, du, x, batch_count, batch_stride, &buffer_size));

#if DEBUG_CUNUMERIC
  assert(buffer_size % 128 == 0);
  std::cout << "buffer size is: " << buffer_size << std::endl;
  std::cout <<"size, count, stride: " << linear_system_size << " " << batch_count << " " << batch_stride << std::endl;
#endif

  // Allocate the buffer
  void* buffer = nullptr;
  if (buffer_size > 0) {
    Legion::DeferredBuffer<char, 1> buf({0, buffer_size - 1}, Memory::GPU_FB_MEM);
    buffer = (void *) buf.ptr(0);
  } 
  
  CHECK_CUSPARSE(gtsv2solver(handle, linear_system_size, dl, d, du, x, batch_count, batch_stride, reinterpret_cast<void *>(buffer)));
  CHECK_CUDA(cudaStreamSynchronize(stream));

}

template<>
struct SolveTridiagonalImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  void operator()(int32_t linear_system_size, float* dl, float* d, float* du, float* x, int32_t batch_count, int32_t batch_stride)
  {
    solve_batched_tridiagonal_template(cusparseSgtsv2StridedBatch_bufferSizeExt,
                       cusparseSgtsv2StridedBatch,
                       linear_system_size,
                       dl, 
                       d, 
                       du, 
                       x, 
                       batch_count,
                       batch_stride);
  }
};

template<>
struct SolveTridiagonalImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  void operator()(int32_t linear_system_size, double* dl, double* d, double* du, double* x, int32_t batch_count, int32_t batch_stride)
  {
    solve_batched_tridiagonal_template(cusparseDgtsv2StridedBatch_bufferSizeExt,
                       cusparseDgtsv2StridedBatch,
                       linear_system_size, 
                       dl, 
                       d, 
                       du, 
                       x, 
                       batch_count,
                       batch_stride);
  }
};

/*

template<>
struct SolveTridiagonalImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  void operator()(int32_t linear_system_size, complex<float>* dl, complex<float>* d, complex<float>* du, complex<float>* x, int32_t batch_count, int32_t batch_stride)
  {
    solve_batched_tridiagonal_template(cusparseCgtsv2_bufferSizeExt,
                       cusparseCgtsv2,
                       linear_system_size, 
                       reinterpret_cast<cuComplex*>(dl),
                       reinterpret_cast<cuComplex*>(d),
                       reinterpret_cast<cuComplex*>(du),
                       reinterpret_cast<cuComplex*>(x),
                       batch_count,
                       batch_stride);

  }
};

template<>
struct SolveTridiagonalImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  void operator()(int32_t linear_system_size, complex<double>* dl, complex<double>* d, complex<double>* du, complex<double>* x, int32_t batch_count, int32_t batch_stride)
  {
    solve_batched_tridiagonal_template(cusparseZgtsv2_bufferSizeExt,
                       cusparseZgtsv2,
                       linear_system_size, 
                       reinterpret_cast<cuDoubleComplex*>(dl),
                       reinterpret_cast<cuDoubleComplex*>(d),
                       reinterpret_cast<cuDoubleComplex*>(du),
                       reinterpret_cast<cuDoubleComplex*>(x),
                       batch_count,
                       batch_stride);
  }
};

*/

/*static*/ void SolveTridiagonalTask::gpu_variant(TaskContext& context)
{
  solve_tridiagonal_template<VariantKind::GPU>(context);
}

} // namespace cunumeric


