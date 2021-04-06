#include "backprop_layer.hpp"
#include <stdexcept>
#include <string>
#include <cmath>

template <typename T, typename C>
BackPropLayer<T, C>::BackPropLayer(const C & num_inputs,
                                   const C & num_ouputs,
                                   const std::shared_ptr<ActivationFunction<T>> & activ_func,
                                   const bool & is_bias_needed):
  Layer<T, C>(num_inputs, num_ouputs, activ_func, is_bias_needed),
  inputs_(0),
  delta_weights_(this->weights_.size())
{
  for (C i = 0; i < this->weights_.size(); ++i)
  {
    delta_weights_[i].resize(this->num_out_);
  }
}

template <typename T, typename C>
BackPropLayer<T, C>::BackPropLayer(const BackPropLayer<T, C> & other):
  Layer<T, C>(other),
  inputs_(other.inputs_),
  delta_h_(other.delta_h_),
  delta_weights_(other.delta_weights_)
{
}

template <typename T, typename C>
BackPropLayer<T, C>& BackPropLayer<T, C>::operator=(const BackPropLayer<T, C> & other)
{
  if (this != &other)
  {
    Layer<T, C>::operator = (other);
    this->inputs_ = other.inputs_;
    this->delta_h_ = other.delta_h_;
    this->delta_weights_ = other.delta_weights_;
  }
  return *this;
}

template <typename T, typename C>
BackPropLayer<T, C>::~BackPropLayer() = default;

template <typename T, typename C>
Vector1D<T> BackPropLayer<T, C>::getInputs() const
{
  return inputs_;
}

template <typename T, typename C>
void BackPropLayer<T, C>::setInputs(const Vector1D<T> & inputs)
{
  if (inputs.size() != this->num_inp_)
  {
    std::string err_msg = "[BackPropLayer - setInputs] Given input vector" \
      " size doesn't equal to specified [" + std::to_string(inputs.size())
        + " != " + std::to_string(this->num_inp_) + "]";
    throw std::invalid_argument(err_msg);
  }

  inputs_ = inputs;
}

template <typename T, typename C>
Vector1D<T> BackPropLayer<T, C>::getDeltas() const
{
  return delta_h_;
}

template <typename T, typename C>
void BackPropLayer<T, C>::calculateDeltas(const Vector1D<T> & prev_delta)
{
  if (prev_delta.size() != this->num_out_)
  {
    std::string err_msg = "[BackPropLayer - calculateDeltas] Given output vector" \
      " size doesn't equal to specified [" + std::to_string(prev_delta.size())
        + " != " + std::to_string(this->num_out_) + "]";
    throw std::invalid_argument(err_msg);
  }

  // Calculate derivatives for neurons outputs
  delta_h_ = inputs_;
  this->activ_func_->derivative(delta_h_);

  // Multiply derivatives with weighted sum
  for (C i = 0; i < this->num_inp_; ++i)
  {
    T sum = 0;

    for (C k = 0; k < this->num_out_; ++k)
    {
      sum += this->weights_[i][k] * prev_delta[k];
    }

    delta_h_[i] *= sum;
  }
}

template <typename T, typename C>
void BackPropLayer<T, C>::applyDerivativeToValues(Vector1D<T> & values) const
{
  this->activ_func_->derivative(values);
}

template <typename T, typename C>
void BackPropLayer<T, C>::recalculateWeights(const T & learn_rate, const T & momentum,
                                          const Vector1D<T> & next_delta_h)
{
  if (next_delta_h.size() != this->num_out_)
  {
    std::string err_msg = "[BackPropLayer - recalculateWeights] Given output vector" \
      " size doesn't equal to specified [" + std::to_string(next_delta_h.size())
        + " != " + std::to_string(this->num_out_) + "]";
    throw std::invalid_argument(err_msg);
  }

  for (C i = 0; i < this->weights_.size(); ++i)
  {
    for (C k = 0; k < this->num_out_; ++k)
    {
      T grad = ((i != this->num_inp_) ? inputs_[i] : 1) * next_delta_h[k];
      delta_weights_[i][k] = learn_rate * grad + momentum * delta_weights_[i][k];
      this->weights_[i][k] += delta_weights_[i][k];
    }
  }
}
