#ifndef NN_COMPARISON_BACKPROP_NETWORK_HPP
#define NN_COMPARISON_BACKPROP_NETWORK_HPP

#include "../base/base_network.hpp"
#include "../../common/dataset/dataset_type.hpp"
#include "../../common/params/backprop_params.hpp"
#include "../../common/vector_types.hpp"
#include "../../common/loss_functions.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
class BackPropNetwork : public BaseNetwork<T, C>
{
public:
  BackPropNetwork(const backprop_params_t<T, C> & params,
                  const std::shared_ptr<LossFunction<T, C>> & loss_function,
                  const bool & is_biases_needed);
  BackPropNetwork(const BackPropNetwork & other);
  BackPropNetwork(BackPropNetwork && other) = delete;
  BackPropNetwork& operator=(const BackPropNetwork & other);
  BackPropNetwork& operator=(BackPropNetwork && other) = delete;
  ~BackPropNetwork() override;
  void buildNetwork() override;
  void train(const dataset_type<T, C> & dataset) override;
  T getBlow() const;
  T getBup() const;
  T getLearnRate() const;
  T getMomentum() const;

private:
  void generateWeights();
  Vector1D<T> predict(const Vector1D<T> & inputs);
  void backpropagation(const Vector1D<T> & inputs, const Vector1D<T> & outputs,
                       const Vector1D<std::size_t> & exp_outputs);

  T b_lo_;
  T b_up_;
  T learn_rate_;
  T momentum_;
};

template class BackPropNetwork<float>;
template class BackPropNetwork<double>;

#endif // NN_COMPARISON_BACKPROP_NETWORK_HPP
