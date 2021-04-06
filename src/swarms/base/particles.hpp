#ifndef NN_COMPARISON_PARTICLES_HPP
#define NN_COMPARISON_PARTICLES_HPP

#include "../../common/vector_types.hpp"

template <typename T>
struct particles_t
{
  Vector1D<T> x;
  Vector1D<T> p;
  Vector1D<T> v;
  Vector1D<T> f_curr;
  Vector1D<T> f_best;
};

#endif // NN_COMPARISON_PARTICLES_HPP
