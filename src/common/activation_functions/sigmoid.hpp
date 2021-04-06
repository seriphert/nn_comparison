#ifndef NN_COMPARISON_SIGMOID_HPP
#define NN_COMPARISON_SIGMOID_HPP

#include "activation_function.hpp"
#include "../../common/vector_types.hpp"

template <typename T>
class Sigmoid : public ActivationFunction<T>
{
public:
  ~Sigmoid() override;
  void calculate(Vector1D<T> & args) const override;
  void derivative(Vector1D<T> & args) const override;
};

template class Sigmoid<float>;
template class Sigmoid<double>;

#endif //NN_COMPARISON_SIGMOID_FUNCTION_HPP
