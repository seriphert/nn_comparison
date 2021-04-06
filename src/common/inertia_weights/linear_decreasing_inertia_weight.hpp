#ifndef NN_COMPARISON_LINEAR_DECREASING_INERTIA_WEIGHT_HPP
#define NN_COMPARISON_LINEAR_DECREASING_INERTIA_WEIGHT_HPP

#include "inertia_weight.hpp"
#include <cstddef>

template <typename T, typename C = std::size_t>
class LinearDecreasingInertiaWeight : public InertiaWeight<T, C>
{
public:
  LinearDecreasingInertiaWeight(const T & w0, const T & w1);
  ~LinearDecreasingInertiaWeight() override;
  T operator()(const C & epoch, const C & n_epochs) override;
  T getW0() const;
  T getW1() const;

private:
  T w0_;
  T w1_;
};

template class LinearDecreasingInertiaWeight<float>;
template class LinearDecreasingInertiaWeight<double>;

#endif //NN_COMPARISON_LINEAR_DECREASING_INERTIA_WEIGHT_HPP
