#ifndef NN_COMPARISON_LAYER_HPP
#define NN_COMPARISON_LAYER_HPP

#include "../../common/vector_types.hpp"
#include "../../common/activation_functions/activation_function.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
class Layer
{
public:
  Layer(const C & num_inputs, const C & num_ouputs,
        const std::shared_ptr<ActivationFunction<T>> & activ_func,
        const bool & is_bias_needed);
  Layer(const Layer & other);
  Layer(Layer && other) = delete;
  Layer& operator=(const Layer & other);
  Layer& operator=(Layer && other) = delete;
  virtual ~Layer();
  Vector1D<T> predict(const Vector1D<T> & inputs) const;
  Vector1D<T> & getWeightsRow(const C & index);
  Vector2D<T> getWeights() const;
  C getNumInputs() const;
  C getNumOutputs() const;

protected:
  C num_inp_;
  C num_out_;
  Vector2D<T> weights_;
  std::shared_ptr<ActivationFunction<T>> activ_func_;
};

template class Layer<float>;
template class Layer<double>;

# endif // NN_COMPARISON_LAYER_HPP
