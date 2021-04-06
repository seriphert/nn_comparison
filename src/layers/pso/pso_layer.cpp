#include "pso_layer.hpp"

template <typename T, typename C>
PSO_Layer<T, C>::PSO_Layer(const C & num_inputs, const C & num_ouputs,
                        const std::shared_ptr<ActivationFunction<T>> & activ_func,
                        const bool & is_bias_needed):
  Layer<T, C>(num_inputs, num_ouputs, activ_func, is_bias_needed)
{
}

template <typename T, typename C>
PSO_Layer<T, C>::PSO_Layer(const PSO_Layer<T, C> & other):
  Layer<T, C>(other)
{
}

template <typename T, typename C>
PSO_Layer<T, C>& PSO_Layer<T, C>::operator=(const PSO_Layer<T, C> & other)
{
  if (this != &other)
  {
    Layer<T, C>::operator=(other);
  }
  return *this;
}

template <typename T, typename C>
PSO_Layer<T, C>::~PSO_Layer() = default;

template <typename T, typename C>
Vector1D<T> PSO_Layer<T, C>::predict(const Vector1D<T> & inputs,
                                  typename Vector1D<T>::const_iterator startWeight) const
{
  Vector1D<T> outputs;

  for (C k = 0; k < this->num_out_; ++k)
  {
    T tmp = 0;

    for (C i = 0; i < this->weights_.size(); ++i)
    {
       // Check if i-th neuron - bias neuron (has 1 as an output to next layer)
      tmp += ((i <this-> num_inp_) ? inputs[i] : 1)
            * (*(startWeight + i * this->num_out_ + k));
    }

    outputs.emplace_back(tmp);
  }

  this->activ_func_->calculate(outputs);
  return outputs;
}
