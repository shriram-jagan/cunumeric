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
/*
template <>
struct support_solve<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_solve<Type::Code::COMPLEX128> : std::true_type {};
*/

template <VariantKind KIND>
struct SolveTridiagonalImpl  {
  template <Type::Code CODE, std::enable_if_t<support_solve<CODE>::value>* = nullptr>
    void operator()(Array& dl_array, Array& d_array, Array& du_array, Array& B_array, int ldb) const
    {
      using VAL = legate_type_of<CODE>;

      const auto dl_shape = dl_array.shape<1>();
      const auto d_shape  = d_array.shape<1>();
      const auto du_shape = du_array.shape<1>();
      const auto b_shape  = B_array.shape<1>();

      // SJ TODO: Make sure ldb is max(1, m)
      // SJ TODO: Let's assume B is of shape (n, 1)

      int m = d_shape.hi[0] - d_shape.lo[0] + 1;
      int n = 1;

      /// int n = b_shape.hi[0] - b_shape.lo[0] + 1; // this is wrong; n is col
      // SJ TODO: This needs to be updated; n reflects the number of columns in B 
      
      size_t dl_strides[1], d_strides[1], du_strides[1], b_strides[1];
      // dl_array.read_accessor<>() would give const VAL *dl; const correctness needs to be 
      // maintained
      VAL *dl = dl_array.read_write_accessor<VAL, 1>(dl_shape).ptr(dl_shape, dl_strides);
      VAL *d  =  d_array.read_write_accessor<VAL, 1>(d_shape).ptr(d_shape  , d_strides);
      VAL *du = du_array.read_write_accessor<VAL, 1>(du_shape).ptr(du_shape, du_strides);

      // SJ TODO: this should change if B_array.dim() == 1
      VAL* b = B_array.read_write_accessor<VAL, 1>(b_shape).ptr(b_shape, b_strides);

      SolveTridiagonalImplBody<KIND, CODE>{}(m, n, dl, d, du, b, ldb);
    }

    template <Type::Code CODE, std::enable_if_t<!support_solve<CODE>::value>* = nullptr>
    void operator()(Array& dl_array, Array& d_array, Array& du_array, Array& B_array, int ldb) const
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
  auto& B  = context.outputs()[3];
  int ldb  = context.scalars()[0].value<int>();

  type_dispatch(d.code(), SolveTridiagonalImpl<KIND>{}, dl, d, du, B, ldb);
}

} // namespace cunumeric
