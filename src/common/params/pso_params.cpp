#include "pso_params.hpp"

template <typename T, typename C>
pso_params_t<T, C>::pso_params_t():
  base_pso_params_t<T>(),
  soc_weight(0)
{
}

template <typename T, typename C>
pso_params_t<T, C>::pso_params_t(const C & numParticles, const T & bLow,
                                 const T & bUp,
                                 const T & cogWeight,
                                 const T & socWeight,
                                 const C & numIters,
                                 const std::shared_ptr<InertiaWeight<T>> & weightFunc):
  base_pso_params_t<T, C>(numParticles, bLow, bUp, cogWeight, numIters, weightFunc),
  soc_weight(socWeight)
{
}

template <typename T, typename C>
pso_params_t<T, C>::~pso_params_t() = default;
