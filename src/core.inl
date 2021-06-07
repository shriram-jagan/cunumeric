/* Copyright 2021 NVIDIA Corporation
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

namespace legate {
namespace numpy {

template <typename T, int DIM>
AccessorRO<T, DIM> RegionField::read_accessor() const
{
  if (transform_.exists())
    return dim_dispatch(
      transform_.shape().first, read_trans_accesor_fn<T, DIM>{}, pr_, fid_, transform_);
  else {
#ifdef LEGION_BOUNDS_CHECKS
    assert(DIM == dim());
#endif
    return AccessorRO<T, DIM>(pr_, fid_);
  }
}

template <typename T, int DIM>
AccessorWO<T, DIM> RegionField::write_accessor() const
{
  if (transform_.exists())
    return dim_dispatch(
      transform_.shape().first, write_trans_accesor_fn<T, DIM>{}, pr_, fid_, transform_);
  else {
#ifdef LEGION_BOUNDS_CHECKS
    assert(DIM == dim());
#endif
    return AccessorWO<T, DIM>(pr_, fid_);
  }
}

template <typename T, int DIM>
AccessorRW<T, DIM> RegionField::read_write_accessor() const
{
  if (transform_.exists())
    return dim_dispatch(
      transform_.shape().first, read_write_trans_accesor_fn<T, DIM>{}, pr_, fid_, transform_);
  else {
#ifdef LEGION_BOUNDS_CHECKS
    assert(DIM == dim());
#endif
    return AccessorRW<T, DIM>(pr_, fid_);
  }
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> RegionField::reduce_accessor() const
{
  if (transform_.exists())
    return dim_dispatch(transform_.shape().first,
                        reduce_trans_accesor_fn<OP, EXCLUSIVE, DIM>{},
                        pr_,
                        fid_,
                        transform_,
                        redop_id_);
  else {
#ifdef LEGION_BOUNDS_CHECKS
    assert(DIM == dim());
#endif
    return AccessorRD<OP, EXCLUSIVE, DIM>(pr_, fid_, OP::REDOP_ID);
  }
}

template <typename T, int DIM>
AccessorRO<T, DIM> RegionField::read_accessor(const Legion::Rect<DIM> &bounds) const
{
  if (transform_.exists())
    return dim_dispatch(
      transform_.shape().first, read_trans_accesor_fn<T, DIM>{}, pr_, fid_, transform_, bounds);
  else {
#ifdef LEGION_BOUNDS_CHECKS
    assert(DIM == dim());
#endif
    return AccessorRO<T, DIM>(pr_, fid_, bounds);
  }
}

template <typename T, int DIM>
AccessorWO<T, DIM> RegionField::write_accessor(const Legion::Rect<DIM> &bounds) const
{
  if (transform_.exists())
    return dim_dispatch(
      transform_.shape().first, write_trans_accesor_fn<T, DIM>{}, pr_, fid_, transform_, bounds);
  else {
#ifdef LEGION_BOUNDS_CHECKS
    assert(DIM == dim());
#endif
    return AccessorWO<T, DIM>(pr_, fid_, bounds);
  }
}

template <typename T, int DIM>
AccessorRW<T, DIM> RegionField::read_write_accessor(const Legion::Rect<DIM> &bounds) const
{
  if (transform_.exists())
    return dim_dispatch(transform_.shape().first,
                        read_write_trans_accesor_fn<T, DIM>{},
                        pr_,
                        fid_,
                        transform_,
                        bounds);
  else {
#ifdef LEGION_BOUNDS_CHECKS
    assert(DIM == dim());
#endif
    return AccessorRW<T, DIM>(pr_, fid_, bounds);
  }
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> RegionField::reduce_accessor(const Legion::Rect<DIM> &bounds) const
{
  if (transform_.exists())
    return dim_dispatch(transform_.shape().first,
                        reduce_trans_accesor_fn<OP, EXCLUSIVE, DIM>{},
                        pr_,
                        fid_,
                        transform_,
                        redop_id_,
                        bounds);
  else {
#ifdef LEGION_BOUNDS_CHECKS
    assert(DIM == dim());
#endif
    return AccessorRD<OP, EXCLUSIVE, DIM>(pr_, fid_, OP::REDOP_ID, bounds);
  }
}

template <int32_t DIM>
Legion::Rect<DIM> RegionField::shape() const
{
  assert(DIM == pr_.get_logical_region().get_index_space().get_dim());
  return Legion::Rect<DIM>(pr_);
}

template <typename T, int DIM>
AccessorRO<T, DIM> Array::read_accessor() const
{
  if (is_future_) {
    auto memkind = Legion::Memory::Kind::NO_MEMKIND;
    return AccessorRO<T, DIM>(future_, memkind, sizeof(T), false, false, NULL, sizeof(uint64_t));
  } else
    return region_field_.read_accessor<T, DIM>();
}

template <typename T, int DIM>
AccessorWO<T, DIM> Array::write_accessor() const
{
  assert(!is_future_);
  return region_field_.write_accessor<T, DIM>();
}

template <typename T, int DIM>
AccessorRW<T, DIM> Array::read_write_accessor() const
{
  assert(!is_future_);
  return region_field_.read_write_accessor<T, DIM>();
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> Array::reduce_accessor() const
{
  assert(!is_future_);
  return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>();
}

template <typename T, int DIM>
AccessorRO<T, DIM> Array::read_accessor(const Legion::Rect<DIM> &bounds) const
{
  if (is_future_) {
    auto memkind = Legion::Memory::Kind::NO_MEMKIND;
    return AccessorRO<T, DIM>(
      future_, bounds, memkind, sizeof(T), false, false, NULL, sizeof(uint64_t));
  } else
    return region_field_.read_accessor<T, DIM>(bounds);
}

template <typename T, int DIM>
AccessorWO<T, DIM> Array::write_accessor(const Legion::Rect<DIM> &bounds) const
{
  assert(!is_future_);
  return region_field_.write_accessor<T, DIM>(bounds);
}

template <typename T, int DIM>
AccessorRW<T, DIM> Array::read_write_accessor(const Legion::Rect<DIM> &bounds) const
{
  assert(!is_future_);
  return region_field_.read_write_accessor<T, DIM>(bounds);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> Array::reduce_accessor(const Legion::Rect<DIM> &bounds) const
{
  assert(!is_future_);
  return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>(bounds);
}

template <int32_t DIM>
Legion::Rect<DIM> Array::shape() const
{
  assert(!is_future_);
  return region_field_.shape<DIM>();
}

}  // namespace numpy
}  // namespace legate