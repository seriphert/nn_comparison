#ifndef NN_COMPARISON_MEAN_SQUARED_ERROR_HPP
#define NN_COMPARISON_MEAN_SQUARED_ERROR_HPP

#include "loss_function.hpp"
#include "../../common/vector_types.hpp"
#include <cstddef>

template <typename T, typename C = std::size_t>
class MeanSquaredError : public LossFunction<T, C>
{
public:
  ~MeanSquaredError() override;
  T calculate(const Vector2D<C> & expected,
              const Vector2D<T> & actual) override;
};

template class MeanSquaredError<float>;
template class MeanSquaredError<double>;

#endif //NN_COMPARISON_MEAN_SQUARED_ERROR_HPP
