#ifndef NN_COMPARISON_ICLPSO_NETWORK_HPP
#define NN_COMPARISON_ICLPSO_NETWORK_HPP

#include "../clpso/clpso_network.hpp"
#include "../../common/params/clpso_params.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
class ICLPSO_Network : public CLPSO_Network<T>
{
public:
  ICLPSO_Network(const clpso_params_t<T, C> & params,
                 const std::shared_ptr<LossFunction<T, C>> & loss_function,
                 const bool & is_biases_needed);
  ICLPSO_Network(const ICLPSO_Network & other);
  ICLPSO_Network(ICLPSO_Network && other) = delete;
  ICLPSO_Network& operator=(const ICLPSO_Network & other);
  ICLPSO_Network& operator=(ICLPSO_Network && other) = delete;
  ~ICLPSO_Network() override;
  void train(const dataset_type<T, C> & dataset) override;
};

template class ICLPSO_Network<float>;
template class ICLPSO_Network<double>;

#endif // NN_COMPARISON_ICLPSO_NETWORK_HPP
