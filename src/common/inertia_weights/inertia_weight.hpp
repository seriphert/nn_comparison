#ifndef NN_COMPARISON_INERTIA_WEIGHT_HPP
#define NN_COMPARISON_INERTIA_WEIGHT_HPP

#include <cstddef>

template <typename T, typename C = std::size_t>
class InertiaWeight
{
public:
  virtual ~InertiaWeight() = 0;
  virtual T operator()(const C & epoch,
                       const C & n_epochs) = 0;
};

template class InertiaWeight<float>;
template class InertiaWeight<double>;

#endif //NN_COMPARISON_INERTIA_WEIGHT_HPP
