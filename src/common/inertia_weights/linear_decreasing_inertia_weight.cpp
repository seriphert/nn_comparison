#include "linear_decreasing_inertia_weight.hpp"

template <typename T, typename C>
LinearDecreasingInertiaWeight<T, C>::LinearDecreasingInertiaWeight(const T & w0,
                                                                   const T & w1):
  w0_(w0),
  w1_(w1)
{
}

template <typename T, typename C>
LinearDecreasingInertiaWeight<T, C>::~LinearDecreasingInertiaWeight() = default;

template <typename T, typename C>
T LinearDecreasingInertiaWeight<T, C>::operator()(const C & epoch,
                                                  const C & n_epochs)
{
  return w0_ - (w0_ - w1_) * static_cast<double>(epoch + 1) / n_epochs;
}

template <typename T, typename C>
T LinearDecreasingInertiaWeight<T, C>::getW0() const
{
  return w0_;
}

template <typename T, typename C>
T LinearDecreasingInertiaWeight<T, C>::getW1() const
{
  return w1_;
}
