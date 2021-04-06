#ifndef NN_COMPARISON_PSO_PARAMS_HPP
#define NN_COMPARISON_PSO_PARAMS_HPP

#include "base_pso_params.hpp"
#include "../inertia_weights.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
struct pso_params_t : base_pso_params_t<T, C>
{
  pso_params_t();
  pso_params_t(const C & numParticles, const T & bLow,
               const T & bUp,
               const T & cogWeight,
               const T & socWeight,
               const C & numIters,
               const std::shared_ptr<InertiaWeight<T>> & weightFunc);
  ~pso_params_t() override;

  T soc_weight;
};

template struct pso_params_t<float>;
template struct pso_params_t<double>;

#endif //NN_COMPARISON_PSO_PARAMS_HPP
