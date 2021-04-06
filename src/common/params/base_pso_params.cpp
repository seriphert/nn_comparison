#include "base_pso_params.hpp"

template <typename T, typename C>
base_pso_params_t<T, C>::base_pso_params_t():
  num_particles(0),
  b_lo(0),
  b_up(0),
  cog_weight(0),
  num_iters(0),
  weight_func(nullptr)
{
}

template <typename T, typename C>
base_pso_params_t<T, C>::base_pso_params_t(const C & numParticles, const T & bLow,
                                           const T & bUp,
                                           const T & cogWeight,
                                           const C & numIters,
                                           const std::shared_ptr<InertiaWeight<T>> & weightFunc):
  num_particles(numParticles),
  b_lo(bLow),
  b_up(bUp),
  cog_weight(cogWeight),
  num_iters(numIters),
  weight_func(weightFunc)
{
}

template <typename T, typename C>
base_pso_params_t<T, C>::~base_pso_params_t() = default;
