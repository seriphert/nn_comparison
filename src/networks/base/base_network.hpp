#ifndef NN_COMPARISON_BASE_NETWORK_HPP
#define NN_COMPARISON_BASE_NETWORK_HPP

#include "../../layers/base/layer.hpp"
#include "../../common/dataset/dataset_type.hpp"
#include "../../common/vector_types.hpp"
#include "../../common/loss_functions/loss_function.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
class BaseNetwork
{
public:
  BaseNetwork(const C & num_epochs,
              const std::shared_ptr<LossFunction<T, C>> & loss_function,
              const bool & is_biases_needed);
  BaseNetwork(const BaseNetwork & other);
  BaseNetwork(BaseNetwork && other) = delete;
  BaseNetwork& operator=(const BaseNetwork & other);
  BaseNetwork& operator=(BaseNetwork && other) = delete;
  virtual ~BaseNetwork() = 0;
  virtual void train(const dataset_type<T, C> & dataset) = 0;
  T accuracy(const dataset_type<T, C> & dataset) const;
  void addLayer(const std::shared_ptr<Layer<T, C>> & layer);
  virtual void buildNetwork() = 0;
  Vector1D<T> predict(const Vector1D<T> & inputs) const;
  Vector2D<T> predict(const Vector2D<T> & inputs_vec) const;
  Vector3D<T> getWeights() const;
  C getNumEpochs() const;
  C getNumLayers() const;
  bool isBuilded() const;

protected:
  // Additional bias for each layer
  C add_bias_;
  bool is_builded_;

  // Network main parameters
  C num_inp_;
  C num_out_;
  C num_epochs_;
  Vector1D<std::shared_ptr<Layer<T, C>>> layers_;
  std::shared_ptr<LossFunction<T, C>> loss_function_;
};

template class BaseNetwork<float>;
template class BaseNetwork<double>;

#endif // NN_COMPARISON_BASE_NETWORK_HPP
