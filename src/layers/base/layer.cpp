#include "layer.hpp"

template <typename T, typename C>
Layer<T, C>::Layer(const C & num_inputs, const C & num_ouputs,
                   const std::shared_ptr<ActivationFunction<T>> & activ_func,
                   const bool & is_bias_needed):
  num_inp_(num_inputs),
  num_out_(num_ouputs),
  activ_func_(activ_func),
  weights_(num_inp_ + ((is_bias_needed) ? 1 : 0))
{
  for (C i = 0; i < weights_.size(); ++i)
  {
    weights_[i].resize(num_out_);
  }
}

template <typename T, typename C>
Layer<T, C>::Layer(const Layer<T, C> & other):
  num_inp_(other.num_inp_),
  num_out_(other.num_out_),
  activ_func_(other.activ_func_),
  weights_(other.weights_)
{
}

template <typename T, typename C>
Layer<T, C>& Layer<T, C>::operator=(const Layer<T, C> & other)
{
  if (this != &other)
  {
    this->num_inp_ = other.num_inp_;
    this->num_out_ = other.num_out_;
    this->activ_func_ = other.activ_func_;
    this->weights_ = other.weights_;
  }
  return *this;
}

template <typename T, typename C>
Layer<T, C>::~Layer() = default;

template <typename T, typename C>
Vector1D<T> Layer<T, C>::predict(const Vector1D<T> & inputs) const
{
  Vector1D<T> outputs;

  for (C k = 0; k < num_out_; ++k)
  {
    T tmp = 0;

    for (C i = 0; i < weights_.size(); ++i)
    {
      // Check if i-th neuron - bias neuron (has 1 as an output)
      tmp += ((i < num_inp_) ? inputs[i] : 1) * weights_[i][k];
    }

    outputs.emplace_back(tmp);
  }

  activ_func_->calculate(outputs);
  return outputs;
}

template <typename T, typename C>
Vector2D<T> Layer<T, C>::getWeights() const
{
  return weights_;
}

template <typename T, typename C>
Vector1D<T> & Layer<T, C>::getWeightsRow(const C & index)
{
  return weights_[index];
}

template <typename T, typename C>
C Layer<T, C>::getNumInputs() const
{
  return num_inp_;
}

template <typename T, typename C>
C Layer<T, C>::getNumOutputs() const
{
  return num_out_;
}
