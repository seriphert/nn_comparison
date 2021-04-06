#ifndef NN_COMPARISON_BACKPROP_LAYER_HPP
#define NN_COMPARISON_BACKPROP_LAYER_HPP

#include "../base/layer.hpp"
#include "../../common/vector_types.hpp"
#include "../../common/activation_functions/activation_function.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
class BackPropLayer : public Layer<T>
{
public:
  BackPropLayer(const C & num_inputs, const C & num_ouputs,
                const std::shared_ptr<ActivationFunction<T>> & activ_func,
                const bool & is_bias_needed);
  BackPropLayer(const BackPropLayer & other);
  BackPropLayer(BackPropLayer && other) = delete;
  BackPropLayer& operator=(const BackPropLayer & other);
  BackPropLayer& operator=(BackPropLayer && other) = delete;
  ~BackPropLayer() override;
  Vector1D<T> getInputs() const;
  void setInputs(const Vector1D<T> & inputs);
  Vector1D<T> getDeltas() const;
  void calculateDeltas(const Vector1D<T> & prev_delta);
  void applyDerivativeToValues(Vector1D<T> & values) const;
  void recalculateWeights(const T & learn_rate, const T & momentum,
                          const Vector1D<T> & next_delta_h);

private:
  Vector1D<T> inputs_;
  Vector1D<T> delta_h_;
  Vector2D<T> delta_weights_;
};

template class BackPropLayer<float>;
template class BackPropLayer<double>;

# endif // NN_COMPARISON_BACKPROP_LAYER_HPP
