#ifndef NN_COMPARISON_CLPSO_NETWORK_HPP
#define NN_COMPARISON_CLPSO_NETWORK_HPP

#include "../base_pso/base_pso_network.hpp"
#include "../../common/params/clpso_params.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
class CLPSO_Network : public BasePSO_Network<T>
{
public:
  CLPSO_Network(const clpso_params_t<T, C> & params,
                const std::shared_ptr<LossFunction<T, C>> & loss_function,
                const bool & is_biases_needed);
  CLPSO_Network(const CLPSO_Network & other);
  CLPSO_Network(CLPSO_Network && other) = delete;
  CLPSO_Network& operator=(const CLPSO_Network & other);
  CLPSO_Network& operator=(CLPSO_Network && other) = delete;
  ~CLPSO_Network();
  void train(const dataset_type<T, C> & dataset) override;
  T getRefreshingGap() const;
};

template class CLPSO_Network<float>;
template class CLPSO_Network<double>;

#endif // NN_COMPARISON_CLPSO_NETWORK_HPP
