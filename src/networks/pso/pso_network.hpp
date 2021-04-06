#ifndef NN_COMPARISON_PSO_NETWORK_HPP
#define NN_COMPARISON_PSO_NETWORK_HPP

#include "../base_pso/base_pso_network.hpp"
#include "../../common/params/pso_params.hpp"
#include "../../common/loss_functions/loss_function.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
class PSO_Network : public BasePSO_Network<T, C>
{
public:
  PSO_Network(const pso_params_t<T, C> & params,
              const std::shared_ptr<LossFunction<T, C>> & loss_function,
              const bool & is_biases_needed);
  PSO_Network(const PSO_Network & other);
  PSO_Network(PSO_Network && other) = delete;
  PSO_Network& operator=(const PSO_Network & other);
  PSO_Network& operator=(PSO_Network && other) = delete;
  ~PSO_Network() override;
  void train(const dataset_type<T, C> & dataset) override;
  T getSocWeight() const;
};

template class PSO_Network<float>;
template class PSO_Network<double>;

#endif // NN_COMPARISON_PSO_NETWORK_HPP
