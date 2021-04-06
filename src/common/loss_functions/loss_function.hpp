#ifndef NN_COMPARISON_LOSS_FUNCTION_HPP
#define NN_COMPARISON_LOSS_FUNCTION_HPP

#include "../../common/vector_types.hpp"
#include <cstddef>

template <typename T, typename C = std::size_t>
class LossFunction
{
public:
  virtual ~LossFunction() = 0;
  virtual T calculate(const Vector2D<C> & expected,
                      const Vector2D<T> & actual) = 0;
};

template class LossFunction<float>;
template class LossFunction<double>;

#endif //NN_COMPARISON_LOSS_FUNCTION_HPP
