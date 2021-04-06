#ifndef NN_COMPARISON_BASE_PSO_NETWORK_HPP
#define NN_COMPARISON_BASE_PSO_NETWORK_HPP

#include "../base/base_network.hpp"
#include "../../common/params/base_pso_params.hpp"
#include "../../common/dataset/dataset_type.hpp"
#include "../../common/vector_types.hpp"
#include "../../common/inertia_weights/inertia_weight.hpp"
#include "../../common/loss_functions/loss_function.hpp"
#include "../../swarms/base/swarm.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
class BasePSO_Network : public BaseNetwork<T, C>
{
public:
  BasePSO_Network(const C & num_iters,
                  const std::shared_ptr<InertiaWeight<T, C>> & weight_func,
                  const std::shared_ptr<LossFunction<T, C>> & loss_function,
                  const bool & is_biases_needed);
  BasePSO_Network(const BasePSO_Network & other);
  BasePSO_Network(BasePSO_Network && other) = delete;
  BasePSO_Network& operator=(const BasePSO_Network & other);
  BasePSO_Network& operator=(BasePSO_Network && other) = delete;
  ~BasePSO_Network() override = 0;
  void buildNetwork() override;
  void train(const dataset_type<T, C> & dataset) override = 0;
  C getNumParticles() const;
  C getNumDimensions() const;
  T getCogWeight() const;
  T getBlo() const;
  T getBup() const;

protected:
  void setWeights(typename Vector1D<T>::const_iterator startWeightsIterator);
  Vector1D<T> predict(const Vector1D<T> & inputs,
                      typename Vector1D<T>::const_iterator weight_iter) const;
  Vector2D<T> predict(const Vector2D<T> & inputs_vec,
                      typename Vector1D<T>::const_iterator weight_iter) const;

  dataset_type<T, C> dataset_;
  std::shared_ptr<Swarm<T, C>> swarm_;
  std::shared_ptr<InertiaWeight<T, C>> weight_func_;
};

template class BasePSO_Network<float>;
template class BasePSO_Network<double>;

#endif // NN_COMPARISON_BASE_PSO_NETWORK_HPP
