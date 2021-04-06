#ifndef NN_COMPARISON_ACTIVATION_FUNCTION_HPP
#define NN_COMPARISON_ACTIVATION_FUNCTION_HPP

#include "../../common/vector_types.hpp"

template <typename T>
class ActivationFunction
{
public:
  virtual ~ActivationFunction() = 0;
  virtual void calculate(Vector1D<T> & args) const = 0;
  virtual void derivative(Vector1D<T> & args) const = 0;
};

template class ActivationFunction<float>;
template class ActivationFunction<double>;

#endif //NN_COMPARISON_ACTIVATION_FUNCTION_HPP
