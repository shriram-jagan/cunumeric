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

#pragma once

#include<vector>

// Useful for IDEs
#include "cunumeric/matrix/solve_tridiagonal.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct SolveTridiagonalImplBody;

template <Type::Code CODE>
struct support_solve : std::false_type {};
template <>
struct support_solve<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_solve<Type::Code::FLOAT32> : std::true_type {};

template <VariantKind KIND>
struct SolveTridiagonalImpl  {
  template <Type::Code CODE, std::enable_if_t<support_solve<CODE>::value>* = nullptr>
    void operator()(Array& dl_array, Array& d_array, Array& du_array, Array& x_array, int batch_count, int batch_stride) const
    {
      using VAL = legate_type_of<CODE>;

      const auto dl_shape = dl_array.shape<1>();
      const auto d_shape  = d_array.shape<1>();
      const auto du_shape = du_array.shape<1>();
      const auto x_shape  = x_array.shape<1>();

      // cusparse APIs use m and n interchangably to denote the size
      // of linear system.
      int linear_system_size = batch_stride;
      
      size_t dl_strides[1], d_strides[1], du_strides[1], x_strides[1];

      VAL *dl = dl_array.read_write_accessor<VAL, 1>(dl_shape).ptr(dl_shape, dl_strides);
      VAL *d  =  d_array.read_write_accessor<VAL, 1>(d_shape).ptr(d_shape  , d_strides);
      VAL *du = du_array.read_write_accessor<VAL, 1>(du_shape).ptr(du_shape, du_strides);

      VAL* x = x_array.read_write_accessor<VAL, 1>(x_shape).ptr(x_shape, x_strides);

      SolveTridiagonalImplBody<KIND, CODE>{}(linear_system_size, dl, d, du, x, batch_count, batch_stride);
    }

    template <Type::Code CODE, std::enable_if_t<!support_solve<CODE>::value>* = nullptr>
    void operator()(Array& dl_array, Array& d_array, Array& du_array, Array& x_array, int batch_count, int batch_stride) const
    {
      assert(false);
    }
};

template <VariantKind KIND>
static void solve_tridiagonal_template(TaskContext &context)
{
  auto& dl = context.outputs()[0];
  auto& d  = context.outputs()[1];
  auto& du = context.outputs()[2];
  auto& x  = context.outputs()[3];
  int batch_count = context.scalars()[0].value<int>();
  int batch_stride = context.scalars()[1].value<int>();

  type_dispatch(d.code(), SolveTridiagonalImpl<KIND>{}, dl, d, du, x, batch_count, batch_stride);
}

} // namespace cunumeric
